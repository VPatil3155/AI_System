
import uuid

import streamlit as st
from langgraph.types import Command

from main import graph


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

    if not snapshot.interrupts:
        raise ValueError(
            "The graph is not waiting at a human approval "
            "interrupt. Do not resume this checkpoint."
        )

    graph.invoke(
        Command(
            resume={
                "decision": decision,
                "feedback": feedback,
            }
        ),
        config=config,
    )

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

    with st.expander("🏗️ Proposed technical design", expanded=False):
        if st.session_state["design"]:
            st.markdown(st.session_state["design"])
        else:
            st.info("The proposed design is not available yet.")

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

    feedback = st.text_area(
        "Would you like to change anything?",
        placeholder=(
            "For example: Add email notifications, make the "
            "app easier to use, or include another feature."
        ),
        height=110,
        key="human_feedback_input",
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
                resume_workflow("APPROVED", feedback.strip())

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
            st.error("We couldn't continue the workflow.")
            st.exception(error)

    if revision_clicked:
        if not feedback.strip():
            st.warning(
                "Please tell us what you would like to change."
            )
        else:
            try:
                with st.spinner(
                    "Updating the design based on your feedback..."
                ):
                    resume_workflow(
                        "NEEDS_REVISION",
                        feedback.strip(),
                    )

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
                st.error("We couldn't update the design.")
                st.exception(error)


# --------------------------------------------------
# IMPLEMENTATION PLAN
# --------------------------------------------------

if st.session_state["workflow_completed"]:
    st.divider()
    st.subheader("📋 Your Implementation Plan")

    st.success(
        "The design has been approved and the AI has prepared "
        "the implementation plan."
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
