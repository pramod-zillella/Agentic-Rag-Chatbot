import streamlit as st
import os
import getpass
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm
from collections import defaultdict
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
import time
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY_V2")

# Authenticate with Hugging Face
huggingface_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if huggingface_token:
    login(huggingface_token)
    
llm = ChatOpenAI(model="gpt-4o-mini")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "fitness-chatbot-enhanced"
index = pc.Index(index_name)
embed_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve detailed fitness information, including exercises, nutrition, and injury prevention strategies, based on the user’s input."""
    query_embedding = embed_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    chunk_ids = set()
    video_ids = set()

    for match in results.matches:
        chunk_id = match['id']
        video_id, chunk_info = chunk_id.split('_', 1)
        chunk_number, total_chunks = chunk_info.split(' of ')

        if video_id not in video_ids:
            video_ids.add(video_id)
            for i in range(1, int(total_chunks) + 1):
                chunk_ids.add(f"{video_id}_{i} of {total_chunks}")
        
        if len(video_ids) >= 3:
            break

    chunk_ids_list = list(chunk_ids)

    results = index.fetch(ids=chunk_ids_list)
    video_chunks = defaultdict(list)

    for chunk_id, chunk_data in  results.vectors.items():
        video_id = chunk_data['metadata']['video_id']
        chunk_content = chunk_data['metadata']['content']
        thumbnail_url = chunk_data['metadata'].get('thumbnail_url', None)
        title = chunk_data['metadata'].get('title', None)
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        video_chunks[video_id].append((chunk_id, chunk_content, thumbnail_url, video_url, title))

    complete_transcripts = {}

    for video_id, chunks in video_chunks.items():
        chunks.sort(key=lambda x: int(x[0].split('_')[1].split(' of ')[0]))
        complete_transcript = " ".join(chunk_content for _, chunk_content, _, _, _ in chunks)
        
        thumbnail_url = chunks[0][2]
        video_url = chunks[0][3]
        title = chunks[0][4]
        
        complete_transcripts[video_id] = {
            "transcript": complete_transcript,
            "thumbnail_url": thumbnail_url,
            "video_url": video_url,
            "title": title
        }
    final_context = []
    for video_id, data in complete_transcripts.items():
        final_context.append(f"{data['transcript']}")
    final_context_document = "\n\n".join(final_context)
        
    return final_context_document, complete_transcripts


def query_or_respond(state: MessagesState):
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    all_transcripts = {}

    docs_content = ""
    for doc in tool_messages:
        docs_content += doc.content
        if doc.artifact:
          all_transcripts.update(doc.artifact)

    system_message_content = f"""
    You are a professional fitness and workout assistant specializing in providing evidence-based advice tailored to beginners. You assist with:
    - Exercise recommendations for different fitness goals (e.g., muscle gain, weight loss, flexibility).
    - Nutrition advice, including meal planning and supplementation.
    - Safety and injury prevention during workouts.

    CONTEXT PROCESSING:
    - First analyze all provided context thoroughly before responding
    - Prioritize information directly from the provided context
    - When multiple pieces of context are relevant, synthesize them coherently
    - If context is insufficient for the query, acknowledge the limitation

    RESPONSE PRINCIPLES:
    - Always ground your responses in the provided context
    - Use clear, jargon-free language
    - Break down complex concepts into digestible steps
    - Emphasize proper form and technique
    - Include relevant safety disclaimers
    - Provide actionable, bite-sized recommendations

    RESPONSE STRUCTURE:
    - Begin with a direct answer to the user's question
    - Support answers with specific references from the context
    - Provide practical implementation steps
    - Include relevant safety considerations
    - End with clear next steps or recommendations

    KNOWLEDGE BOUNDARIES:
    - Only provide advice based on available context
    - Clearly distinguish between general principles and specific recommendations
    - If medical advice is needed, direct users to healthcare professionals
    - Acknowledge when a question requires expertise beyond the provided context

    QUERY HANDLING:
    - Consider user's fitness level when providing recommendations
    - Ensure all advice is suitable for beginners

    KNOWLEDGE-BASE CONTEXT: 
    {docs_content}
        """
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)

    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>AthleanX Fitness</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-weight: bold;'>Ask me about workouts, nutrition, or injury prevention!</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Here are a few common fitness questions:</p>", unsafe_allow_html=True)
        
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What's the best way to prevent age-related muscle loss?"):
            st.session_state.current_question = "What's the best way to prevent age-related muscle loss?"
        if st.button("How can I prevent wrist pain during push-ups and planks?"):
            st.session_state.current_question = "How can I prevent wrist pain during push-ups and planks?"
    with col2:
        if st.button("I sit at a desk all day. What exercises can help with posture?"):
            st.session_state.current_question = "I sit at a desk all day. What exercises can help with posture?"
        if st.button("Can you suggest a full-body workout routine for beginners?"):
            st.session_state.current_question = "Can you suggest a full-body workout routine for beginners?"
    st.markdown("<p style='text-align: center; font-weight: bold; color: #27AE60;'>Expert advice in under 10 seconds!</p>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = None

def add_message(role, content, recommendations=None, response_time=None):
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "recommendations": recommendations,
        "response_time": response_time
    })

def display_chat_history():
    """
    Displays the chat history in the correct chronological order.
    New messages are appended to the chat history, and the UI is updated dynamically.
    """
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            message_placeholder = st.empty()
            message_placeholder.markdown(message["content"], unsafe_allow_html=True)

            if message.get("recommendations"):
                st.subheader("Video Recommendations:")
                cols = st.columns(min(3, len(message["recommendations"])))

                for idx, (video_id, data) in enumerate(list(message["recommendations"].items())[:3]):
                    with cols[idx]:
                        st.image(data['thumbnail_url'], use_container_width=True)
                        st.write(f"**{data['title']}**")
                        st.markdown(f"[Watch Video]({data['video_url']})")

async def stream_response(prompt):
    start_time = time.time()
    with st.chat_message("assistant"):  
        with st.spinner("✨ Gathering fitness insights... ✨"):
            message_placeholder = st.empty()
            # tools_placeholder = st.empty()  # Placeholder for tool updates
            full_response = ""
            all_transcripts = {}
            # tool_updates = []

            async for message, metadata in graph.astream(
                {"messages": [HumanMessage(content=prompt)]},
                stream_mode="messages",  # Stream all messages
            ):
                # Handle only AI-generated responses
                if not isinstance(message, HumanMessage) and message.type != "tool":
                    full_response += message.content
                    message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    # await asyncio.sleep(0.05)

                # # Display tool call information
                # if message.type == "tool":
                #     tool_updates.append(f"Tool Called: {message.tool_name}")
                #     tools_placeholder.markdown(
                #         "<br>".join(tool_updates), 
                #         unsafe_allow_html=True
                #     )

                # Store transcripts if tool generates artifacts
                if message.type == "tool" and message.artifact:
                    all_transcripts.update(message.artifact)

            # Final AI response display
            message_placeholder.markdown(full_response.strip(), unsafe_allow_html=True)

            # Display recommendations
            if all_transcripts:
                st.subheader("Video Recommendations:")
                cols = st.columns(min(3, len(all_transcripts)))
                for idx, (video_id, data) in enumerate(list(all_transcripts.items())[:3]):
                    if idx >= 3:
                        break
                    with cols[idx]:
                        st.image(data['thumbnail_url'], use_container_width=True)
                        st.write(f"**{data['title']}**")
                        st.markdown(f"[Watch Video]({data['video_url']})")
    end_time = time.time()
    response_time = end_time - start_time
    return full_response, all_transcripts, response_time

if st.session_state.current_question:
    add_message("user", st.session_state.current_question)
    with st.chat_message("user"):
        st.markdown(st.session_state.current_question)
    response, recommendations, response_time = asyncio.run(
        stream_response(st.session_state.current_question)
    )
    add_message("assistant", response, recommendations, response_time)
    st.session_state.current_question = None
    st.rerun()

if prompt := st.chat_input("What's your fitness or nutrition question?"):
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    response, recommendations, response_time = asyncio.run(
        stream_response(prompt)
    )
    add_message("assistant", response, recommendations, response_time)
    st.rerun()

display_chat_history()
