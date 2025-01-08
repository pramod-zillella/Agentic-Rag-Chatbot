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

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY_V2")

llm = ChatOpenAI(model="gpt-4o-mini")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "fitness-chatbot-enhanced"
index = pc.Index(index_name)
embed_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve fitness-related context and information based on the user’s query."""
    query_embedding = embed_model.encode(query).tolist()
    # Query Pinecone index for the top 5 results
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # To store chunk_ids and track distinct video_ids
    chunk_ids = set()  # Set ensures no duplicates
    video_ids = set()  # To track distinct videos

    # Iterate through the matches and collect chunks for distinct videos
    for match in results['matches']:
        chunk_id = match['id']
        video_id, chunk_info = chunk_id.split('_', 1)
        chunk_number, total_chunks = chunk_info.split(' of ')

        if video_id not in video_ids:
            # Add the video_id and the corresponding chunks
            video_ids.add(video_id)
            for i in range(1, int(total_chunks) + 1):
                chunk_ids.add(f"{video_id}_{i} of {total_chunks}")
        
        # Stop once we've collected chunks from 3 distinct videos
        if len(video_ids) >= 3:
            break

    chunk_ids_list = list(chunk_ids)

    # Fetch the vectors from Pinecone using chunk_ids_list
    results = index.fetch(ids=chunk_ids_list)

    # A dictionary to store video chunks by video_id, along with metadata
    video_chunks = defaultdict(list)

    # Iterate through the vectors and group chunks by video_id
    for chunk_id, chunk_data in results['vectors'].items():
        # Extract video_id and chunk content from the metadata
        video_id = chunk_data['metadata']['video_id']
        chunk_content = chunk_data['metadata']['content']
        thumbnail_url = chunk_data['metadata'].get('thumbnail_url', None)  # Optional field
        title = chunk_data['metadata'].get('title', None)
        # Construct video_url
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Append the chunk content and additional metadata to the list for this video_id
        video_chunks[video_id].append((chunk_id, chunk_content, thumbnail_url, video_url, title))

    # Now, we need to reconstruct the full transcript for each video
    complete_transcripts = {}

    for video_id, chunks in video_chunks.items():
        # Sort chunks by chunk_number without needing to split multiple times
        chunks.sort(key=lambda x: int(x[0].split('_')[1].split(' of ')[0]))
        # Concatenate the chunks to form the complete transcript
        complete_transcript = " ".join(chunk_content for _, chunk_content, _, _, _ in chunks)
        
        # Get the thumbnail_url from the first chunk for the video
        thumbnail_url = chunks[0][2]  # Assuming all chunks for the video have the same thumbnail URL
        video_url = chunks[0][3]      # Assuming all chunks for the video have the same video URL
        title = chunks[0][4]          # Assuming all chunks for the video have the same title
        
        # Store the full transcript and metadata in a dictionary for this video_id
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


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """
    Retrieve fitness-related context and information when:
    - User asks about specific exercises
    - Requests workout routines
    - Needs form guidance
    - Asks about nutrition
    - Requires safety information
    
    Args:
        query (str): The user's fitness-related query
    Returns:
        tuple: (context_document, detailed_transcripts)
    """
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Initialize an empty dictionary to store all the complete_transcripts
    all_transcripts = {}

    docs_content = ""
    # Format into prompt and store all complete_transcripts
    for doc in tool_messages:
        docs_content += doc.content
        if doc.artifact:
          all_transcripts.update(doc.artifact)

    system_message_content = f"""
    You are a professional fitness and workout assistant powered by authenticated fitness knowledge. Your primary purpose is to assist beginners in their fitness journey by providing evidence-based advice.

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
    # Run
    response = llm.invoke(prompt)

    # Now you can access all_transcripts here for downstream tasks
    # print("Complete transcripts:", all_transcripts)

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
##############################################################################################################
# Streamlit UI 
with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>AthleanX Fitness</h2>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: center;'>Ask me anything about workouts, nutrition, or injury prevention!</h6>", unsafe_allow_html=True)
        st.write("<p style='text-align: center;'>To get you started here are a few common fitness questions:</p>", unsafe_allow_html=True)

# Dynamic quadrant layout using Streamlit's columns
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("What's the best way to prevent age-related muscle loss?")
        st.markdown("How can I prevent wrist pain during push-ups and planks?")
    with col2:
        st.markdown("I sit at a desk all day. What exercises can help with posture?")
        st.markdown("Can you suggest a full-body workout routine for beginners?")
        # if st.button("What's a good workout split for building muscle?"):
        #     st.session_state.current_question = "What's a good workout split for building muscle?"

# Initialize chat history in session state
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
  Displays the chat history in reverse order, excluding the latest interaction.

  This function iterates through the chat history stored in st.session_state 
  and displays each message within a chat message container. It also includes 
  video recommendations (if available) for each message.
  """
  pairs = [st.session_state.chat_history[i:i+2] for i in range(0, len(st.session_state.chat_history[:-2]), 2)]
    # -> [[11, 12], [21, 22], [31, 32]]
    # Reverse the pairs and flatten
  result = [x for pair in reversed(pairs) for x in pair]
  for message in result:
    with st.chat_message(message["role"]):
      message_placeholder = st.empty()
      message_placeholder.markdown(message["content"], unsafe_allow_html=True)

      if message.get("recommendations"):
        st.subheader("Video Recommendations:")
        cols = st.columns(min(3, len(message["recommendations"])))

        for idx, (video_id, data) in enumerate(message["recommendations"].items()):
          with cols[idx]:
            st.image(data['thumbnail_url'], use_container_width=True)
            st.write(f"**{data['title']}**")
            st.markdown(f"[Watch Video]({data['video_url']})")
            
def stream_response(prompt):
    start_time = time.time()
    with st.chat_message("assistant"):  
        with st.spinner("✨ Gathering fitness insights just for you... ✨"):
            message_placeholder = st.empty()
            # message_placeholder.empty()
            full_response = ""
            final_draft = ""
            all_transcripts = {}

            for step in graph.stream(
                {"messages": [HumanMessage(content=prompt)]},
                stream_mode="values",
            ):
                if "messages" in step and step["messages"][-1].type == "ai":
                    response_content = step["messages"][-1].content
                    final_draft += response_content
                    response_content = response_content.replace('**', '__')
                    response_content = response_content.replace('\n\n', '<br><br>')
                    response_content = response_content.replace('\n', '<br>')

                    for chunk in response_content.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                
                if "messages" in step:
                    for message in step["messages"]:
                        if message.type == "tool" and message.artifact: #Checks if the message is a tool message and if it contains an artifact
                            all_transcripts.update(message.artifact) #If the conditions are true then store the artifact in all_transcripts variable
            message_placeholder.markdown(final_draft, unsafe_allow_html=True) # Final formatted display

            if all_transcripts:
                st.subheader("Video Recommendations:")
                cols = st.columns(min(3, len(all_transcripts)))
                for idx, (video_id, data) in enumerate(all_transcripts.items()):
                    if idx >= 3:
                        break
                    with cols[idx]:
                        st.image(data['thumbnail_url'], use_container_width=True)
                        st.write(f"**{data['title']}**")
                        st.markdown(f"[Watch Video]({data['video_url']})")
    end_time = time.time()
    response_time = end_time - start_time
    return full_response, all_transcripts, response_time

# Process current question from preselected options or user input
if st.session_state.current_question:
    add_message("user", st.session_state.current_question)
    with st.chat_message("user"):
        st.markdown(st.session_state.current_question)
    response, recommendations, response_time = stream_response(st.session_state.current_question)
    add_message("assistant", response, recommendations, response_time)
    st.session_state.current_question = None

# Capture user input to keep the conversation flowing
if prompt := st.chat_input("What's your fitness or nutrition question?"):
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    response, recommendations, response_time = stream_response(prompt)
    add_message("assistant", response, recommendations, response_time)

# Render chat history (once at the end, if needed)
if len(st.session_state.chat_history) > 0:
    display_chat_history()


