import gradio as gr
import asyncio
import json
from datetime import datetime
from scientia.core.knowledge_system import ScientiaCore
from scientia.core.models import KnowledgePacket
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScientiaApp:
    def __init__(self):
        self.core = ScientiaCore()
        self.chat_history = []
        
        # Create exports directory if it doesn't exist
        self.exports_dir = os.path.join(os.getcwd(), "scientia_exports")
        os.makedirs(self.exports_dir, exist_ok=True)
        logger.info(f"Exports will be saved to: {self.exports_dir}")

    async def chat(self, message, history, context=""):
        """Handle chat interactions"""
        try:
            result = await self.core.chat(message, context)
            self.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "response": result["response"],
                "confidence": result["confidence"]
            })
            return result["response"]
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Error: {str(e)}"

    async def add_text_knowledge(self, text, metadata=""):
        """Add text knowledge to the system"""
        try:
            packet = KnowledgePacket(
                content=text,
                metadata=json.loads(metadata) if metadata else {},
                source_type="user_input",
                confidence=1.0
            )
            await self.core.add_to_knowledge_base(packet)
            return f"Successfully added knowledge: {text[:100]}..."
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
            return f"Error: {str(e)}"

    async def import_shared_knowledge(self, file):
        """Import knowledge from a shared file"""
        try:
            with open(file.name, 'r') as f:
                data = json.load(f)
            
            for item in data:
                packet = KnowledgePacket(**item)
                await self.core.add_to_knowledge_base(packet)
            
            return f"Successfully imported {len(data)} knowledge packets"
        except Exception as e:
            logger.error(f"Error importing knowledge: {str(e)}")
            return f"Error: {str(e)}"

    async def export_knowledge(self, topic):
        """Export knowledge about a topic to a file, including both stored and model-generated knowledge"""
        try:
            # Create filename based on topic and timestamp
            filename = f"scientia_export_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.exports_dir, filename)
            
            # Get stored knowledge
            stored_knowledge = await self.core.query_knowledge(topic, k=50)
            
            # Define comprehensive questions about the topic
            questions = [
                f"Please provide a comprehensive overview of {topic}.",
                f"What are the key concepts and fundamental principles of {topic}?",
                f"What is the historical development and evolution of {topic}?",
                f"What are the main applications and practical uses of {topic}?",
                f"What are the current challenges and future developments in {topic}?",
                f"What are the most important discoveries or breakthroughs related to {topic}?",
                f"How does {topic} impact or relate to other fields?",
                f"What are the common misconceptions about {topic}?",
                f"Who are the key figures or organizations involved in {topic}?",
                f"What are the latest research trends in {topic}?"
            ]
            
            # Gather model knowledge from multiple questions
            model_responses = []
            for question in questions:
                response = await self.core.chat(question)
                model_responses.append({
                    "question": question,
                    "response": response["response"],
                    "confidence": response["confidence"]
                })
            
            # Format for export
            export_data = {
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "stored_knowledge": [
                    {
                        "content": item["content"],
                        "metadata": item["metadata"],
                        "source_type": "stored_knowledge",
                        "confidence": item["similarity"]
                    }
                    for item in stored_knowledge
                ],
                "model_knowledge": {
                    "overview": model_responses[0],  # Comprehensive overview
                    "key_concepts": model_responses[1],  # Key concepts
                    "history": model_responses[2],  # Historical development
                    "applications": model_responses[3],  # Applications
                    "challenges_and_future": model_responses[4],  # Challenges and future
                    "breakthroughs": model_responses[5],  # Important discoveries
                    "interdisciplinary_relations": model_responses[6],  # Relations to other fields
                    "misconceptions": model_responses[7],  # Common misconceptions
                    "key_figures": model_responses[8],  # Key figures
                    "research_trends": model_responses[9],  # Latest trends
                    "model": self.core.model.config.name_or_path
                }
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return (f"Exported to: {filepath}\n"
                   f"Included {len(stored_knowledge)} stored knowledge entries and "
                   f"{len(model_responses)} comprehensive model-generated responses about {topic}.")
            
        except Exception as e:
            logger.error(f"Error exporting knowledge: {str(e)}")
            return f"Error: {str(e)}"

    def get_chat_history(self):
        """Get formatted chat history"""
        return "\n\n".join([
            f"User ({h['timestamp']}): {h['message']}\nAssistant (confidence: {h['confidence']:.2f}): {h['response']}"
            for h in self.chat_history
        ])

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Scientia Knowledge System") as interface:
            gr.Markdown("# Scientia Knowledge System")
            
            with gr.Tab("Chat"):
                chatbot = gr.ChatInterface(
                    fn=self.chat,
                    title="Chat with Scientia",
                    description="Ask questions or have a conversation with the knowledge system",
                    examples=[
                        ["What do you know about artificial intelligence?"],
                        ["How does machine learning work?"],
                        ["Tell me about neural networks"]
                    ],
                    additional_inputs=[
                        gr.Textbox(label="Additional Context (optional)", lines=2)
                    ]
                )

            with gr.Tab("Knowledge Management"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Add New Knowledge")
                        text_input = gr.Textbox(label="Text Content", lines=5)
                        metadata_input = gr.Textbox(
                            label="Metadata (JSON format)", 
                            lines=2, 
                            placeholder='{"source": "manual", "tags": ["AI", "ML"]}'
                        )
                        add_btn = gr.Button("Add Knowledge")
                        add_output = gr.Textbox(label="Status")
                        add_btn.click(
                            fn=self.add_text_knowledge,
                            inputs=[text_input, metadata_input],
                            outputs=add_output
                        )

                    with gr.Column():
                        gr.Markdown("### Import Shared Knowledge")
                        file_input = gr.File(label="Import Knowledge File")
                        import_btn = gr.Button("Import")
                        import_output = gr.Textbox(label="Import Status")
                        import_btn.click(
                            fn=self.import_shared_knowledge,
                            inputs=[file_input],
                            outputs=import_output
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Export Knowledge")
                        topic_input = gr.Textbox(label="Topic to Export")
                        export_btn = gr.Button("Export")
                        export_output = gr.Textbox(label="Export Status")
                        export_btn.click(
                            fn=lambda x: asyncio.run(self.export_knowledge(x)),  # Wrap in asyncio.run
                            inputs=[topic_input],
                            outputs=export_output
                        )

            with gr.Tab("Chat History"):
                history_btn = gr.Button("Refresh History")
                history_output = gr.Textbox(label="Chat History", lines=20)
                history_btn.click(
                    fn=self.get_chat_history,
                    inputs=[],
                    outputs=history_output
                )

        return interface

def main():
    app = ScientiaApp()
    interface = app.create_interface()
    interface.launch()

if __name__ == "__main__":
    main()
