
import uuid
import hashlib
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from langgraph.types import Command

from main import graph


_design_selection_component = components.declare_component(
    "design_selection",
    path=str(Path(__file__).parent / "design_selection_component"),
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Software Development System",
    page_icon="🤖",
    layout="wide",
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULTS = {
    "thread_id": None,
    "user_request": "",
    "requirements": "",
    "design": "",
    "ai_review": "",
    "task_plan": "",
    "human_decision": "",
    "human_feedback": "",
    "selected_design_text": "",
    "revision_comment": "",
    "workflow_started": False,
    "waiting_for_human": False,
    "workflow_completed": False,
    "workflow_error": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_config():
    thread_id = st.session_state.get("thread_id")

    if not thread_id:
        raise ValueError("No active workflow was found.")

    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def reset_workflow():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

    st.session_state.pop("human_feedback_input", None)
    st.session_state.pop("revision_comment_input", None)


def sync_graph_state():
    """Synchronize the latest graph checkpoint with the UI."""
    snapshot = graph.get_state(get_config())
    values = snapshot.values

    for key in (
        "user_request",
        "requirements",
        "design",
        "ai_review",
        "task_plan",
        "human_decision",
        "human_feedback",
    ):
        if key in values:
            st.session_state[key] = values[key]

    st.session_state["waiting_for_human"] = bool(snapshot.interrupts)

    task_plan = st.session_state.get("task_plan", "")
    st.session_state["workflow_completed"] = (
        bool(task_plan.strip())
        and not snapshot.interrupts
        and not snapshot.next
    )


def resume_workflow(decision, feedback):
    config = get_config()
    snapshot = graph.get_state(config)

    print("Thread ID:", config["configurable"]["thread_id"])
    print("Checkpoint keys:", list(snapshot.values.keys()))
    print("Next nodes:", snapshot.next)
    print("Pending interrupts:", snapshot.interrupts)

    if "user_request" not in snapshot.values:
        raise ValueError(
            "The checkpoint is missing user_request. "
            "Start a new project and inspect the checkpoint."
        )

    if not snapshot.interrupts or "human_approval" not in snapshot.next:
        raise ValueError(
            "The graph is not waiting at a human approval "
            "interrupt. Do not resume this checkpoint."
        )

    try:
        graph.invoke(
            Command(
                resume={
                    "decision": decision,
                    "feedback": feedback,
                }
            ),
            config=config,
        )
    except Exception:
        # The interrupt is consumed before downstream nodes run. Keep the UI
        # aligned with the checkpoint even if a later node raises, while
        # preserving the original planner/workflow exception.
        try:
            sync_graph_state()
        except Exception as sync_error:
            print("Could not sync graph state after workflow failure:", sync_error)
        raise
    else:
        sync_graph_state()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 AI Software Development System")

st.markdown(
    """
    Turn your idea into software with the help of AI.

    Describe what you want to build in your own words.
    You don't need technical knowledge—we'll help turn
    your idea into a design and implementation plan.
    """
)

st.divider()


# --------------------------------------------------
# IDEA INPUT
# --------------------------------------------------

st.subheader("What would you like to build?")

with st.form("idea_form"):
    user_request = st.text_area(
        "Describe your idea",
        placeholder=(
            "For example:\n\n"
            "I want an app for college students where they "
            "can create projects, find teammates, assign "
            "tasks, and track their progress.\n\n"
            "Just explain your idea naturally. You don't "
            "need to specify technologies or databases."
        ),
        height=180,
        label_visibility="collapsed",
    )

    submitted = st.form_submit_button(
        "🚀 Turn My Idea Into Software",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not user_request.strip():
        st.warning("Please describe your idea first.")
    else:
        reset_workflow()

        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["user_request"] = user_request.strip()
        st.session_state["workflow_started"] = True

        initial_state = {
            "user_request": user_request.strip(),
            "requirements": "",
            "design": "",
            "ai_review": "",
            "human_decision": "",
            "human_feedback": "",
    "selected_design_text": "",
    "revision_comment": "",
            "task_plan": "",
        }

        try:
            with st.spinner(
                "🤖 Understanding your idea and preparing "
                "a proposed design..."
            ):
                graph.invoke(
                    initial_state,
                    config=get_config(),
                )

                sync_graph_state()

            st.rerun()

        except Exception as error:
            st.session_state["workflow_error"] = str(error)
            st.error("We couldn't process your idea.")
            st.exception(error)


# --------------------------------------------------
# WORKFLOW ERROR
# --------------------------------------------------

if st.session_state["workflow_error"]:
    st.error(st.session_state["workflow_error"])


# --------------------------------------------------
# AI-GENERATED RESULTS
# --------------------------------------------------

if st.session_state["workflow_started"]:
    st.divider()
    st.subheader("Your Idea, Understood by AI")

    st.caption(
        "The following sections are generated by AI. "
        "You don't need to fill them in yourself."
    )

    with st.expander("📝 What the AI understood", expanded=False):
        if st.session_state["requirements"]:
            st.markdown(st.session_state["requirements"])
        else:
            st.info("The requirements analysis is not available yet.")

    st.subheader("AI Review")

    if st.session_state["ai_review"]:
        st.markdown(st.session_state["ai_review"])
    else:
        st.info("The design review is not available yet.")


# --------------------------------------------------
# HUMAN APPROVAL
# --------------------------------------------------

if (
    st.session_state["workflow_started"]
    and st.session_state["waiting_for_human"]
    and not st.session_state["workflow_completed"]
):
    st.divider()
    st.subheader("Review the Proposed Design")

    st.write(
        "The AI has prepared a proposal for your idea. "
        "Review it and decide whether to continue or request changes."
    )

    st.markdown(
        "Highlight any text in the design below, then describe the change you want."
    )

    design_text = st.session_state.get("design", "")
    selected_text = _design_selection_component(
        design=design_text,
        key=(
            "design_selection_"
            + str(st.session_state.get("thread_id"))
            + "_"
            + hashlib.sha256(design_text.encode("utf-8")).hexdigest()[:12]
        ),
        default="",
    )
    if isinstance(selected_text, str):
        st.session_state["selected_design_text"] = selected_text

    selected_text = st.session_state.get("selected_design_text", "")
    if selected_text.strip():
        st.caption("Selected text")
        st.code(selected_text, language=None)
    else:
        st.info("Highlight a portion of the design above to select it.")

    revision_comment = st.text_area(
        "What should change about this selection?",
        placeholder=(
            "For example: Replace MongoDB with PostgreSQL because "
            "the application needs relational data and transactions."
        ),
        height=110,
        key="revision_comment_input",
    )

    col1, col2 = st.columns(2)

    with col1:
        approve_clicked = st.button(
            "✅ I Accept",
            use_container_width=True,
            key="approve_design",
        )

    with col2:
        revision_clicked = st.button(
            "🔄 Make Changes",
            use_container_width=True,
            key="revise_design",
        )

    if approve_clicked:
        try:
            with st.spinner("Creating the implementation plan..."):
                resume_workflow("APPROVED", "")

            if st.session_state["workflow_completed"]:
                st.success("Your implementation plan is ready.")
            elif st.session_state["waiting_for_human"]:
                st.info("The updated proposal needs your review.")
            else:
                st.warning(
                    "The workflow resumed, but completion "
                    "has not been confirmed."
                )

            st.rerun()

        except Exception as error:
            st.session_state["workflow_error"] = str(error)
            st.error("We couldn't continue the workflow.")
            st.exception(error)

    if revision_clicked:
        if not selected_text.strip():
            st.warning("Please select a portion of the design first.")
        elif not revision_comment.strip():
            st.warning("Please describe how you want the selected text changed.")
        else:
            revision_feedback = f"""TARGETED DESIGN REVISION

SELECTED DESIGN TEXT:
{selected_text.strip()}

USER COMMENT:
{revision_comment.strip()}

REVISION INSTRUCTIONS:
Revise the selected portion according to the user's comment.
Update other sections only when necessary for consistency.
Preserve unrelated design decisions, requirements, and features.
Return the complete revised design, not only the selected text.
"""
            try:
                with st.spinner(
                    "Updating the design based on your selected text..."
                ):
                    resume_workflow(
                        "NEEDS_REVISION",
                        revision_feedback,
                    )
                    st.session_state["selected_design_text"] = ""
                    st.session_state["revision_comment"] = ""

                if st.session_state["waiting_for_human"]:
                    st.success(
                        "The updated proposal is ready for review."
                    )
                else:
                    st.warning(
                        "The workflow did not return to the expected "
                        "review checkpoint. Check the graph state."
                    )

                st.rerun()

            except Exception as error:
                st.session_state["workflow_error"] = str(error)
                st.error("We couldn't update the design.")
                st.exception(error)


# --------------------------------------------------
# IMPLEMENTATION PLAN
# --------------------------------------------------

if st.session_state["task_plan"].strip():
    st.divider()
    st.subheader("📋 Your Implementation Plan")

    if st.session_state["workflow_completed"]:
        st.success(
            "The design has been approved and the AI has prepared "
            "the implementation plan."
        )
    else:
        st.info(
            "The implementation plan is ready. A later workflow step "
            "did not finish; see the workflow error above."
        )

    st.markdown(st.session_state["task_plan"])

    st.download_button(
        "📥 Download Implementation Plan",
        data=st.session_state["task_plan"],
        file_name="implementation_plan.md",
        mime="text/markdown",
        use_container_width=True,
    )


# --------------------------------------------------
# START A NEW PROJECT
# --------------------------------------------------

if st.session_state["workflow_started"]:
    st.divider()

    if st.button("➕ Start a New Project"):
        reset_workflow()
        st.rerun()
