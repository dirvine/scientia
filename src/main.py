import asyncio
import streamlit as st
from scientia.core.knowledge_system import ScientiaCore, KnowledgePacket
import json
from datetime import datetime
import sys
import hashlib
import torch

# Set page configuration first, before any other Streamlit commands
st.set_page_config(
    page_title="Scientia AI Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"  # Changed from "expanded" to "collapsed"
)

# Check if running under Streamlit
def is_running_under_streamlit():
    try:
        # This will only succeed if running under streamlit
        st.runtime.exists()
        return True
    except:
        return False

if not is_running_under_streamlit():
    print("This is a Streamlit app, please run it with:")
    print("   streamlit run src/main.py")
    sys.exit(1)

# Near the top of the file, after imports
# Initialize the ScientiaCore with better caching
@st.cache_resource(show_spinner="Loading AI models...")
def initialize_scientia():
    """Initialize ScientiaCore with models and keep them in memory"""
    with st.spinner("Loading models... This may take a few minutes on first run."):
        core = ScientiaCore(enable_multimodal=False)
        
        # Warm up the models with a test inference
        # This ensures everything is loaded and ready
        with torch.no_grad():
            test_prompt = "Hello, this is a test."
            inputs = core.tokenizer(test_prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(core.device) for k, v in inputs.items()}
            _ = core.model.generate(
                **inputs,
                max_new_tokens=5,
                num_return_sequences=1
            )
        
        return core

# Initialize scientia instance once and keep it in memory
try:
    scientia = initialize_scientia()
    st.success("✅ AI models loaded and ready!")
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# Remove the old initialization code
# @st.cache_resource
# def get_scientia():
#     return ScientiaCore(enable_multimodal=False)
# scientia = get_scientia()

# Sidebar for global settings
with st.sidebar:
    st.title("🧠 Scientia AI")
    st.markdown("---")
    
    # Global settings
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 
                          help="Higher values make the output more creative but less focused")
    max_length = st.slider("Max Response Length", 100, 1000, 500,
                          help="Maximum length of generated responses")
    
    st.markdown("---")
    st.markdown("### Export Settings")
    auto_export = st.checkbox("Auto-export results", value=True,
                            help="Automatically save results to JSON files")
    
    if auto_export:
        export_path = st.text_input("Export Directory", 
                                  value="scientia_exports",
                                  help="Directory where results will be saved")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs([
    "💬 Chat Interface",
    "📚 Knowledge Base",
    "⚙️ Advanced Tools"
])

# Chat Interface Tab
with tab1:
    # Create a title with help icon
    col1, col2 = st.columns([10, 1])
    with col1:
        st.title("Chat Interface")
    with col2:
        st.markdown("""
        <div class="tooltip" style="text-align: right; padding-top: 15px;">
            <span>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="24" height="24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/>
                </svg>
            </span>
            <div class="tooltiptext">
                This chat interface combines knowledge from our curated knowledge base with the capabilities of our AI model.
                All conversations are stored in memory during your session.<br><br>
                The system will:<br>
                • Search the knowledge base for relevant information<br>
                • Use found knowledge to enhance responses<br>
                • Fall back to the AI model's knowledge when needed<br>
                • Maintain context throughout your conversation
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add custom CSS for the tooltip with improved styling
    st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .tooltip svg {
        color: #555;
        transition: color 0.3s ease;
    }

    .tooltip:hover svg {
        color: #ff4b4b;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #ffffff;
        color: #31333F;
        text-align: left;
        border-radius: 6px;
        padding: 15px;
        position: absolute;
        z-index: 1000;
        top: 130%;
        right: 0;
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #f0f2f6;
        font-size: 0.9em;
        line-height: 1.5;
    }

    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        bottom: 100%;
        right: 10px;
        border-width: 8px;
        border-style: solid;
        border-color: transparent transparent #ffffff transparent;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add topic analysis selector
    analysis_mode = st.radio(
        "Interaction Mode",
        ["Chat", "Topic Analysis"],
        horizontal=True,
        help="Choose between normal chat or comprehensive topic analysis"
    )
    
    # Show topic analysis options if selected
    if analysis_mode == "Topic Analysis":
        with st.expander("Topic Analysis Options", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                max_queries = st.slider(
                    "Exploration Depth",
                    min_value=5,
                    max_value=20,
                    value=10,
                    help="Number of aspects to explore"
                )
            with col2:
                exploration_depth = st.select_slider(
                    "Detail Level",
                    options=["Basic", "Detailed", "Comprehensive"],
                    value="Detailed",
                    help="How detailed should the analysis be"
                )
            
            include_sections = st.multiselect(
                "Include in Analysis",
                ["Historical Context", "Current State", "Future Implications", 
                 "Technical Details", "Practical Applications", "Related Concepts"],
                default=["Current State", "Technical Details", "Practical Applications"],
                help="Select which aspects to include in the analysis"
            )
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Create main chat container
    chat_container = st.container()
    
    # Add New Conversation button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show suggested questions if they exist
                if "suggested_questions" in message:
                    st.markdown("#### Related Questions:")
                    cols = st.columns(2)
                    for i, question in enumerate(message["suggested_questions"]):
                        col_idx = i % 2
                        if cols[col_idx].button(f"🔍 {question}", key=f"q_{i}_{hash(question)}"):
                            st.session_state.messages.append({"role": "user", "content": question})
                            st.rerun()
    
    # Chat input at the bottom
    if prompt := st.chat_input(
        "What would you like to know?" if analysis_mode == "Chat" 
        else "Enter a topic to analyze..."
    ):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            if analysis_mode == "Topic Analysis":
                with st.spinner("Performing comprehensive analysis..."):
                    # Adjust exploration prompt based on selected sections
                    section_prompts = {
                        "Historical Context": "historical development and key milestones",
                        "Current State": "current status, developments, and understanding",
                        "Future Implications": "future potential and implications",
                        "Technical Details": "technical aspects and mechanisms",
                        "Practical Applications": "real-world applications and use cases",
                        "Related Concepts": "related concepts and interconnections"
                    }
                    
                    selected_aspects = [section_prompts[section] for section in include_sections]
                    
                    # Perform full topic analysis with selected options
                    results = asyncio.run(scientia.explore_topic(
                        prompt,
                        max_queries=max_queries
                    ))
                    
                    # Format the comprehensive response based on detail level
                    response_text = f"""
### Comprehensive Analysis: {prompt}

{results['summary']}

#### Key Findings:
"""
                    # Adjust number of findings based on detail level
                    findings_limit = {
                        "Basic": 3,
                        "Detailed": 5,
                        "Comprehensive": 10
                    }[exploration_depth]
                    
                    for finding in results["detailed_findings"][:findings_limit]:
                        response_text += f"\n- **{finding['question']}**\n  {finding['answer'][:300]}...\n"
                    
                    response_text += f"\n\nConfidence: {results['confidence']*100:.1f}%"
                    
                    # Generate follow-up questions focused on selected aspects
                    follow_up_prompt = f"""Based on this analysis of {prompt}, suggest 5 follow-up questions 
                    focusing on these aspects: {', '.join(selected_aspects)}"""
                    follow_up = asyncio.run(scientia.chat(follow_up_prompt))
                    
                    response = {
                        "response": response_text,
                        "confidence": results["confidence"],
                        "knowledge_used": True,
                        "suggested_questions": [q.strip() for q in follow_up['response'].split('\n') if q.strip()]
                    }
            else:  # Normal chat mode
                # Initialize placeholder for streaming output
                message_placeholder = st.empty()
                full_response = ""
                
                # Initialize the response dict
                response = {"confidence": 0.0, "knowledge_used": False, "relevant_knowledge": []}
                
                # First, get relevant knowledge
                with st.spinner("Searching knowledge base..."):
                    relevant_knowledge = asyncio.run(scientia.query_knowledge(prompt))
                    response["relevant_knowledge"] = relevant_knowledge
                    response["knowledge_used"] = bool(relevant_knowledge)
                    if relevant_knowledge:
                        response["confidence"] = max(item['similarity'] for item in relevant_knowledge)
                
                # Format context from relevant knowledge
                knowledge_context = "\n".join([
                    f"Important context: {item['content']}" 
                    for item in relevant_knowledge 
                    if item['similarity'] > 0.5
                ])
                
                # Prepare the prompt
                prompt_template = f"""You are a helpful AI assistant with access to a knowledge base. 
                Use the following verified information from the knowledge base to inform your response:

{knowledge_context}

If the knowledge base contains relevant information, prioritize using it in your response.
If no relevant information is found in the knowledge base, indicate that and provide a general response.

User: {prompt}
Assistant: Let me help you based on the information available."""

                # Tokenize input
                inputs = scientia.tokenizer(prompt_template, return_tensors="pt", padding=True)
                inputs = {k: v.to(scientia.device) for k, v in inputs.items()}
                
                # Generate response with streaming
                with torch.no_grad():
                    outputs = scientia.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        num_return_sequences=1,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=scientia.tokenizer.eos_token_id,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    # Stream the tokens
                    for i in range(len(outputs.sequences[0])):
                        current_text = scientia.tokenizer.decode(outputs.sequences[0][:i+1], skip_special_tokens=True)
                        
                        # Remove the prompt from the generated text
                        if current_text.startswith(prompt_template):
                            current_text = current_text[len(prompt_template):].strip()
                        
                        # Update the display
                        full_response = current_text
                        message_placeholder.markdown(full_response + "▌")
                
                # Final update without the cursor
                message_placeholder.markdown(full_response)
                response["response"] = full_response
                
                # Generate follow-up questions
                with st.spinner("Generating follow-up questions..."):
                    question_prompt = f"Based on the topic '{prompt}' and the response provided, what are 3-4 relevant follow-up questions?"
                    follow_up = asyncio.run(scientia.chat(question_prompt))
                    response["suggested_questions"] = [q.strip() for q in follow_up['response'].split('\n') if q.strip()]
            
            # Show confidence and sources in expander
            with st.expander("View details"):
                st.metric("Response Confidence", 
                         f"{response['confidence']*100:.1f}%")
                
                if response.get("knowledge_used"):
                    st.success("✓ Response includes information from knowledge base")
                    if response.get("relevant_knowledge"):
                        st.markdown("### Relevant Knowledge Used")
                        for source in response["relevant_knowledge"]:
                            if source['similarity'] > 0.5:
                                st.markdown(f"""
                                - Relevance: {source['similarity']:.2f}
                                - Content: {source['content'][:200]}...
                                """)
                else:
                    st.warning("No relevant information found in knowledge base")
        
        # Add assistant response to chat history with suggested questions
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["response"],
            "suggested_questions": response.get("suggested_questions", [])
        })

# Knowledge Base Tab
with tab2:
    st.title("Knowledge Base")
    
    # Create tabs for different operations
    kb_tab1, kb_tab2 = st.tabs(["🔍 Search Knowledge", "➕ Add Knowledge"])
    
    # Search Knowledge Tab
    with kb_tab1:
        st.subheader("Search Knowledge Base")
        
        # Search options
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_query = st.text_input("Search Query", placeholder="Enter your search terms...")
        with search_col2:
            num_results = st.number_input("Max Results", min_value=1, max_value=20, value=5)
        
        # Advanced search filters in expander
        with st.expander("Advanced Filters"):
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                filter_privacy = st.multiselect(
                    "Privacy Levels",
                    ["PUBLIC", "PRIVATE", "RESTRICTED"],
                    default=["PUBLIC"]
                )
            with filter_col2:
                filter_tags = st.text_input(
                    "Filter by Tags",
                    placeholder="tag1, tag2, tag3"
                )
        
        # Search button
        if st.button("🔍 Search", use_container_width=True) and search_query:
            with st.spinner("Searching knowledge base..."):
                # Convert tags to list
                tag_list = [tag.strip() for tag in filter_tags.split(",") if tag.strip()]
                
                # Perform search
                results = asyncio.run(scientia.query_knowledge(
                    search_query,
                    k=num_results
                ))
                
                # Display results
                if results:
                    for result in results:
                        with st.expander(f"Match ({result['similarity']:.2f})", expanded=True):
                            st.markdown(result['content'])
                            if result.get('metadata'):
                                st.caption(f"Tags: {', '.join(result['metadata'].get('tags', []))}")
                                st.caption(f"Added: {result['metadata'].get('added_date', 'Unknown')}")
                else:
                    st.info("No matching results found.")
    
    # Add Knowledge Tab
    with kb_tab2:
        st.subheader("Add to Knowledge Base")
        
        # Common options for both text and file inputs
        with st.expander("Knowledge Settings", expanded=True):
            opt_col1, opt_col2 = st.columns(2)
            with opt_col1:
                privacy_level = st.selectbox(
                    "Privacy Level",
                    ["PUBLIC", "PRIVATE", "RESTRICTED"],
                    help="Control who can access this knowledge"
                )
            with opt_col2:
                knowledge_tags = st.text_input(
                    "Tags (comma-separated)",
                    placeholder="e.g. science, physics, energy",
                    help="Add tags to help organize and find this knowledge"
                )
        
        # Input method selector
        input_method = st.radio(
            "Input Method",
            ["Text Input", "Document Upload"],
            horizontal=True,
            help="Choose how you want to add knowledge"
        )
        
        if input_method == "Text Input":
            # Text input form
            knowledge_input = st.text_area(
                "Enter Knowledge",
                height=200,
                placeholder="Enter the knowledge you want to add..."
            )
            
            if st.button("Add to Knowledge Base", use_container_width=True) and knowledge_input:
                try:
                    with st.spinner("Adding to knowledge base..."):
                        # Create and add knowledge packet
                        knowledge = KnowledgePacket(
                            content=knowledge_input,
                            embeddings=None,
                            source_type="USER_INPUT",
                            timestamp=datetime.now().isoformat(),
                            confidence=0.9,
                            context_hash=hashlib.sha256(knowledge_input.encode()).hexdigest(),
                            privacy_level=privacy_level,
                            metadata={
                                "tags": [tag.strip() for tag in knowledge_tags.split(",") if tag.strip()],
                                "added_by": "user",
                                "added_date": datetime.now().isoformat()
                            }
                        )
                        
                        # Use asyncio.run instead of await
                        asyncio.run(scientia.add_to_knowledge_base(knowledge))
                        st.success("Successfully added to knowledge base!")
                        
                        # Clear the input
                        st.session_state.knowledge_input = ""
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error adding to knowledge base: {str(e)}")
        
        else:  # Document Upload
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'docx', 'doc', 'jpg', 'jpeg', 'png', 'gif'],
                accept_multiple_files=True,
                help="Select one or more files to upload"
            )
            
            if uploaded_files:
                # Show upload options
                with st.form("document_upload_form"):
                    st.markdown("### Process Selected Documents")
                    
                    # Display file list
                    for file in uploaded_files:
                        st.text(f"📄 {file.name}")
                    
                    # Submit button
                    process_button = st.form_submit_button(
                        "Process Documents",
                        use_container_width=True
                    )
                
                if process_button:
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, file in enumerate(uploaded_files):
                            try:
                                status_text.text(f"Processing {file.name}...")
                                
                                # Process document
                                knowledge_packets = asyncio.run(
                                    scientia.process_document(file, file.name)
                                )
                                
                                # Update metadata for each packet
                                for packet in knowledge_packets:
                                    packet.privacy_level = privacy_level
                                    packet.metadata.update({
                                        "tags": [tag.strip() for tag in knowledge_tags.split(",") if tag.strip()],
                                        "added_by": "user",
                                        "added_date": datetime.now().isoformat()
                                    })
                                    
                                    # Add to knowledge base
                                    asyncio.run(scientia.add_to_knowledge_base(packet))
                                
                                progress_bar.progress((i + 1) / len(uploaded_files))
                                
                            except Exception as e:
                                st.error(f"Error processing {file.name}: {str(e)}")
                        
                        status_text.text("Processing complete!")
                        st.success(f"Successfully processed {len(uploaded_files)} documents")

# Advanced Tools Tab
with tab3:
    st.title("Advanced Tools")
    
    tool_selection = st.selectbox(
        "Select Tool",
        ["Knowledge Graph Visualization", "Concept Mapper", "Source Analyzer"]
    )
    
    if tool_selection == "Knowledge Graph Visualization":
        st.info("Knowledge graph visualization coming soon!")
        
    elif tool_selection == "Concept Mapper":
        st.info("Concept mapping tool coming soon!")
        
    elif tool_selection == "Source Analyzer":
        st.info("Source analysis tool coming soon!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Powered by Scientia AI 🧠 | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

