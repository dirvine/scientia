import asyncio
import os
from src.scientia.core.knowledge_system import ScientiaCore
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    try:
        # Initialize Scientia system
        logger.info("Initializing Scientia system...")
        scientia = ScientiaCore()  # Removed token requirement

        # Demo interaction
        while True:
            # Get user input
            user_input = input("\nEnter your question (or 'exit' to quit): ")
            
            if user_input.lower() == 'exit':
                break

            # Process the input as knowledge
            logger.info("Processing input as knowledge...")
            hash_id = await scientia.process_knowledge(user_input)
            
            # Query existing knowledge
            logger.info("Querying knowledge base...")
            context = await scientia.query_knowledge(user_input)
            
            # Generate chat response
            logger.info("Generating response...")
            response = await scientia.chat(user_input, context=context)
            
            print(f"\nResponse: {response}")
            
            # Show related knowledge
            if context:
                print("\nRelated Knowledge:")
                for i, item in enumerate(context, 1):
                    print(f"{i}. {item.content[:100]}...")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
