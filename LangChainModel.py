from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
import os
import json

load_dotenv()

class Topic(BaseModel):
    """Model for extracted topics from syllabus."""
    title: str = Field(description="The title of the topic")
    description: str = Field(description="Brief description of the topic")
    subtopics: Optional[List[str]] = Field(default_factory=list, description="List of subtopics if any")

class Topics(BaseModel):
    """Container for multiple topics."""
    topics: List[Topic] = Field(description="List of topics from the syllabus")

class LocalStudyMaterialRAG:
    def __init__(
        self,
        gemini_api_key: str = os.getenv("GEMINI_API_KEY"),
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_storage_dir: str = "./vector_db"
    ):
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=gemini_api_key,
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        self.syllabus_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100,
            length_function=len
        )
        
        self.reference_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=200,
            length_function=len
        )
        
        # Create local storage directories
        self.local_storage_dir = local_storage_dir
        self.syllabus_dir = os.path.join(local_storage_dir, "syllabus")
        self.reference_dir = os.path.join(local_storage_dir, "references")
        
        os.makedirs(self.syllabus_dir, exist_ok=True)
        os.makedirs(self.reference_dir, exist_ok=True)
        
        # Initialize empty vector stores - will be populated as data is added
        self.syllabus_store = None
        self.reference_store = None
        
        # Initialize empty collections to hold texts and metadata
        self.syllabus_texts = []
        self.syllabus_metadatas = []
        self.reference_texts = []
        self.reference_metadatas = []
        
        # Try to load existing vector stores if available
        self._load_vector_stores()
        
        self.topic_extraction_prompt = PromptTemplate(
            template="""Extract the main topics from this syllabus content. For each topic, provide a title, 
            brief description, and any subtopics mentioned.
            
            Syllabus: {syllabus_content}
            
            Format the output as a JSON list of topics with their descriptions and subtopics.
            {format_instructions}
            """,
            input_variables=["syllabus_content"],
            partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=Topics).get_format_instructions()}
        )

        self.study_material_prompt = PromptTemplate(
            template="""Create comprehensive, information-dense slides in Marp-compatible markdown for:

                        Topic: {topic}
                        Description: {description}

                        Based on these detailed reference materials:
                        {reference_content}

                        Content Requirements:
                        - Create approx 20 slides
                        - Include extensive factual content, definitions, and explanations
                        - Cover the topic thoroughly with academic depth
                        - Include relevant theories, methodologies, historical context, and current applications
                        - Define all technical terminology completely
                        - Incorporate statistics, research findings, and scholarly perspectives
                        - Include key examples that demonstrate practical applications
                        - Provide comprehensive explanations of complex concepts

                        Formatting Guidelines:
                        - Begin each slide with '---'
                        - PLEASE DO NOT ADD ```markdown
                        - Make sure the slide do no overflow 
                        - Use hierarchical headings to organize dense information (## for main titles, ### for subtitles)
                        - Employ multi-level bullet points for detailed breakdowns
                        - Format each slide to maximize information while maintaining readability
                        - Use **bold** and *italics* to highlight critical terms and concepts
                        - ADD reference at the end . Make sure the references are valid and do no use your own brain (which you ofcourse do not have)
                        - ADD Atleast 1 MERMAID DIAGRAMS. Also make sure the diagram is small to fit and does not overflow

                        DO NOT:
                        - Add Marp metadata (I'll handle that separately)
                        - Include image references
                        - Write annotations like "Here is the slide content"
                        - Sacrifice depth for brevity
                        - Omit important details or nuances
                        - ADD ``` markdown code block
                        """,
            input_variables=["topic", "description", "reference_content", "teacher_id"]
        )
    
    def _get_processed_file_path(self, file_type: str, file_name: str) -> str:
        """
        Generate the file path for a processed file.

        Args:
            file_type: Type of the file (e.g., 'syllabus', 'textbook').
            file_name: Name of the file.

        Returns:
            Full path to the processed file.
        """
        return os.path.join(self.local_storage_dir, f"{file_type}_{file_name}_processed.json")

    def _load_vector_stores(self):
        """
        Try to load existing vector stores if available, otherwise initialize empty ones.
        """
        try:
            if os.path.exists(os.path.join(self.syllabus_dir, "index.faiss")):
                self.syllabus_store = FAISS.load_local(
                    self.syllabus_dir, self.embeddings, allow_dangerous_deserialization=True
                )
                print("Loaded existing syllabus vector store")
            
            if os.path.exists(os.path.join(self.reference_dir, "index.faiss")):
                self.reference_store = FAISS.load_local(
                    self.reference_dir, self.embeddings, allow_dangerous_deserialization=True
                )
                print("Loaded existing reference vector store")
        except Exception as e:
            print(f"Error loading existing vector stores: {e}")
            # If loading fails, we'll create new ones when adding data

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file. If PyPDF2 fails, use pdfplumber as a fallback.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text as a string.
        """
        text = ""
        try:
            # Attempt to extract text using PyPDF2
            with open(pdf_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text()

            # If PyPDF2 fails to extract meaningful text, use pdfplumber
            if len(text.strip()) < 100:  # Threshold for minimal text
                print(f"PyPDF2 failed to extract meaningful text. Using pdfplumber for: {pdf_path}")
                with pdfplumber.open(pdf_path) as pdf:
                    text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

            if not text.strip():
                print(f"Error: No text could be extracted from the PDF: {pdf_path}")
            else:
                print(f"Successfully extracted text from {pdf_path}. Extracted {len(text)} characters.")
        except Exception as e:
            print(f"Error reading PDF file {pdf_path}: {e}")
        return text

    def add_syllabus(self, syllabus_text: str, metadata: Dict[str, Any] = None):
        """
        Preprocess and add syllabus document to the local syllabus collection and extract topics.

        Args:
            syllabus_text: Text content of the syllabus
            metadata: Metadata for the syllabus (course name, teacher ID, etc.)
        """
        preprocess_prompt = PromptTemplate(
            template="""Improve the following syllabus content by making it more structured, clear, and concise.
            Ensure the content is well-organized and easy to understand.
            Only add things which look MODULES. DO NOT ADD COURSE OUTCOME, LAB RELATED THINGS. Also just give Modules and units please.

            Original Syllabus:
            {syllabus_content}

            Improved Syllabus:""",
            input_variables=["syllabus_content"]
        )
        chain = preprocess_prompt | self.llm
        result = chain.invoke({"syllabus_content": syllabus_text})

        try:
            improved_syllabus = result.content
            print("Syllabus text improved successfully.")
        except Exception as e:
            print(f"Error improving syllabus text: {e}")
            improved_syllabus = syllabus_text 

        texts = self.syllabus_splitter.split_text(improved_syllabus)

        if not metadata:
            metadata = {}

        if 'teacher_id' not in metadata:
            print("Warning: No teacher_id specified in metadata")
            metadata['teacher_id'] = "local_teacher"

        if 'document_type' not in metadata:
            metadata['document_type'] = 'syllabus'

        metadatas = [metadata.copy() for _ in texts]
        
        # Store texts and metadatas for FAISS
        self.syllabus_texts.extend(texts)
        self.syllabus_metadatas.extend(metadatas)
        
        # Create or update the vector store
        if self.syllabus_store is None:
            self.syllabus_store = FAISS.from_texts(
                texts=self.syllabus_texts,
                embedding=self.embeddings,
                metadatas=self.syllabus_metadatas
            )
        else:
            self.syllabus_store.add_texts(texts=texts, metadatas=metadatas)
        
        # Save the updated vector store
        self.syllabus_store.save_local(self.syllabus_dir)
        print(f"Saved {len(texts)} syllabus chunks to local storage")

        combined_syllabus = "\n\n".join(texts)
        parser = PydanticOutputParser(pydantic_object=Topics)
        chain = self.topic_extraction_prompt | self.llm

        result = chain.invoke({"syllabus_content": combined_syllabus})

        try:
            topics_container = parser.parse(result.content)
            topics = topics_container.topics

            topic_texts = [
                f"Title: {topic.title}\nDescription: {topic.description}\nSubtopics: {', '.join(topic.subtopics or [])}"
                for topic in topics
            ]
            topic_metadatas = [
                {
                    "topic_title": topic.title,
                    "teacher_id": metadata.get("teacher_id", "local_teacher"),
                    "document_type": "extracted_topic"
                }
                for topic in topics
            ]

            # Add topics to the store
            self.syllabus_texts.extend(topic_texts)
            self.syllabus_metadatas.extend(topic_metadatas)
            
            if self.syllabus_store is not None:
                self.syllabus_store.add_texts(texts=topic_texts, metadatas=topic_metadatas)
                self.syllabus_store.save_local(self.syllabus_dir)
                
            # Save extracted topics to a JSON file for easy reference
            topics_json = [
                {
                    "title": topic.title,
                    "description": topic.description,
                    "subtopics": topic.subtopics or []
                }
                for topic in topics
            ]
            
            with open(os.path.join(self.local_storage_dir, "extracted_topics.json"), "w") as f:
                json.dump(topics_json, f, indent=2)
                
            print(f"Extracted topics added to the store: {[topic.title for topic in topics]}")
            print(f"Topics also saved to {os.path.join(self.local_storage_dir, 'extracted_topics.json')}")

        except Exception as e:
            print(f"Error extracting topics: {e}")
            print(f"Raw output: {result.content}")

    def add_reference_material(self, reference_text: str, metadata: Dict[str, Any] = None):
        """
        Add reference material to the local reference collection.

        Args:
            reference_text: Text content of the reference material
            metadata: Metadata for the reference (source, author, teacher ID, etc.)
        """
        if not reference_text.strip():
            print("Error: Reference text is empty. Please provide valid reference material.")
            return

        texts = self.reference_splitter.split_text(reference_text)
        if not texts:
            print("Error: No valid chunks could be generated from the reference text.")
            return

        if not metadata:
            metadata = {}

        if 'teacher_id' not in metadata:
            print("Warning: No teacher_id specified in metadata")
            metadata['teacher_id'] = "local_teacher"

        if 'document_type' not in metadata:
            metadata['document_type'] = 'reference'

        metadatas = [metadata.copy() for _ in texts]

        # Ensure embeddings are generated before proceeding
        try:
            embeddings = self.embeddings.embed_documents(texts)
        except AttributeError as e:
            print(f"Error: Failed to generate embeddings for the reference text. {e}")
            return

        # Store texts and metadatas for FAISS
        self.reference_texts.extend(texts)
        self.reference_metadatas.extend(metadatas)

        # Create or update the vector store
        if self.reference_store is None:
            self.reference_store = FAISS.from_texts(
                texts=self.reference_texts,
                embedding=self.embeddings,
                metadatas=self.reference_metadatas
            )
        else:
            self.reference_store.add_texts(texts=texts, metadatas=metadatas)

        # Save the updated vector store
        self.reference_store.save_local(self.reference_dir)
        print(f"Saved {len(texts)} reference chunks to local storage")

    def _load_processed_file(self, file_path: str) -> Optional[Dict]:
        """
        Load a processed file if it exists.

        Args:
            file_path: Path to the processed file.

        Returns:
            The loaded data as a dictionary, or None if the file does not exist.
        """
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_processed_file(self, file_path: str, data: Dict):
        """
        Save data to a processed file.

        Args:
            file_path: Path to the processed file.
            data: Data to save.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Processed file saved to: {file_path}")

    def process_textbook(self, pdf_path: str) -> Dict:
        """
        Process a textbook PDF to extract and structure content.

        Args:
            pdf_path: Path to the textbook PDF.

        Returns:
            Structured content as a dictionary.
        """
        file_name = os.path.basename(pdf_path).replace(".pdf", "")
        processed_file_path = self._get_processed_file_path("textbook", file_name)

        # Check if the textbook is already processed
        processed_data = self._load_processed_file(processed_file_path)
        if processed_data:
            print(f"Processed textbook already exists: {processed_file_path}")
            return processed_data

        # Extract and structure content
        print(f"Processing textbook: {pdf_path}")
        raw_text = self.extract_text_from_pdf(pdf_path)
        chapters = raw_text.split("\n\nChapter")  # Example heuristic for splitting chapters
        structured_content = {
            f"Chapter {i+1}": chapter.strip() for i, chapter in enumerate(chapters) if chapter.strip()
        }

        # Save the processed textbook
        self._save_processed_file(processed_file_path, structured_content)
        return structured_content

    def process_syllabus(self, syllabus_text: str) -> Dict:
        """
        Process a syllabus to extract and structure content.

        Args:
            syllabus_text: Text content of the syllabus.

        Returns:
            Structured content as a dictionary.
        """
        processed_file_path = self._get_processed_file_path("syllabus", "syllabus")

        # Check if the syllabus is already processed
        processed_data = self._load_processed_file(processed_file_path)
        if processed_data:
            print(f"Processed syllabus already exists: {processed_file_path}")
            return processed_data

        # Extract and structure content
        print("Processing syllabus...")
        self.add_syllabus(syllabus_text, {"document_type": "syllabus"})
        structured_content = {
            "syllabus_texts": self.syllabus_texts,
            "syllabus_metadatas": self.syllabus_metadatas,
        }

        # Save the processed syllabus
        self._save_processed_file(processed_file_path, structured_content)
        return structured_content

    def generate_slides_from_textbook_and_syllabus(self, topic_query: str, textbook_pdf: str, syllabus_text: str, teacher_id: str = "local_teacher") -> str:
        """
        Generate slides based on a topic by processing the textbook and syllabus.

        Args:
            topic_query: The topic to search for.
            textbook_pdf: Path to the textbook PDF.
            syllabus_text: Text content of the syllabus.
            teacher_id: ID of the teacher creating the slides.

        Returns:
            Path to the generated markdown file.
        """
        # Process the textbook and syllabus
        textbook_content = self.process_textbook(textbook_pdf)
        syllabus_content = self.process_syllabus(syllabus_text)

        # Search for the topic in the syllabus
        print(f"Searching syllabus for topic: {topic_query}")
        topics = self.extract_topics(topic_query)
        if not topics:
            print(f"No relevant topics found in the syllabus for: {topic_query}")
            return f"No relevant topics found for: {topic_query}"

        # Use the first relevant topic
        topic = topics[0]
        print(f"Found topic: {topic.title}")

        # Search for the topic in the textbook
        print(f"Searching textbook for topic: {topic_query}")
        relevant_chapters = [
            chapter for chapter, content in textbook_content.items() if topic_query.lower() in content.lower()
        ]
        reference_content = "\n\n".join([textbook_content[chapter] for chapter in relevant_chapters])
        if not reference_content.strip():
            print(f"No relevant content found in the textbook for: {topic_query}")
            return f"No relevant content found in the textbook for: {topic_query}"

        # Generate study material
        print(f"Generating study material for topic: {topic.title}")
        chain = self.study_material_prompt | self.llm
        result = chain.invoke({
            "topic": topic.title,
            "description": topic.description,
            "reference_content": reference_content,
            "teacher_id": teacher_id
        })

        # Save the study material to a markdown file
        output_dir = os.path.join(self.local_storage_dir, "generated_ppts")
        os.makedirs(output_dir, exist_ok=True)

        file_name = f"{topic.title.replace(' ', '_').lower()}_slides.md"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write('''---\nmarp: true\ntheme: gaia\npaginate: true\nbackgroundColor: "#1E1E2E"\ncolor: white\n\n''')
            f.write(result.content)

        print(f"PPT-compatible markdown saved to: {file_path}")

        # Convert markdown to PPTX using Marp CLI
        pptx_path = file_path.replace(".md", ".pptx")
        try:
            os.system(f"marp {file_path} --pptx -o {pptx_path}")
            print(f"PPTX file generated: {pptx_path}")
        except Exception as e:
            print(f"Failed to convert markdown to PPTX: {e}")
            return f"Markdown saved, but PPTX conversion failed: {file_path}"

        return pptx_path

    def create_full_course_materials(self, course_query: str, teacher_id: str = "local_teacher") -> Dict[str, str]:
        """
        Create study materials for a single topic based on syllabus and references.

        Args:
            course_query: Query to find the relevant syllabus topic
            teacher_id: ID of the teacher creating the course

        Returns:
            Dictionary mapping the topic title to its study material content
        """
        # Step 1: Extract topics from the syllabus
        print(f"Extracting topics for course query: {course_query}")
        topics = self.extract_topics(course_query)
        if not topics:
            print("No topics could be extracted from the syllabus.")
            return {"error": "No topics could be extracted from the syllabus."}

        # Use the first relevant topic
        topic = topics[0]
        print(f"Found topic: {topic.title}")

        # Step 2: Generate study material for the topic
        print(f"Generating study material for topic: {topic.title}")
        material = self.generate_study_material(topic, teacher_id)

        # Save the material to a local file
        output_dir = os.path.join(self.local_storage_dir, "generated_materials")
        os.makedirs(output_dir, exist_ok=True)

        file_name = f"{topic.title.replace(' ', '_').lower()}_material.md"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(material)

        print(f"Study material for '{topic.title}' saved to {file_path}")

        return {topic.title: material}

    def extract_topics(self, course_query: str) -> List[Topic]:
        """
        Extract topics from the syllabus based on a course query.

        Args:
            course_query: Query to find relevant syllabus content.

        Returns:
            List of Topic objects.
        """
        if self.syllabus_store is None:
            print("No syllabus data available. Please add a syllabus first.")
            return []

        # Perform similarity search in the syllabus vector store
        results = self.syllabus_store.similarity_search(course_query, k=3)
        if not results:
            print("No relevant syllabus content found.")
            return []

        syllabus_content = "\n\n".join([doc.page_content for doc in results])

        # Parse the topics using the topic extraction prompt
        parser = PydanticOutputParser(pydantic_object=Topics)
        chain = self.topic_extraction_prompt | self.llm

        result = chain.invoke({"syllabus_content": syllabus_content})

        try:
            topics_container = parser.parse(result.content)
            return topics_container.topics
        except Exception as e:
            print(f"Error parsing topics: {e}")
            print(f"Raw output: {result.content}")
            return []

    def generate_study_material(self, topic: Topic, teacher_id: str = "local_teacher") -> str:
        """
        Generate study material for a specific topic.

        Args:
            topic: Topic object with title, description, and subtopics.
            teacher_id: ID of the teacher creating the material.

        Returns:
            Marp-compatible markdown for the study material.
        """
        if self.reference_store is None:
            print("No reference materials available. Please add reference materials first.")
            return f"# {topic.title}\n\nNo reference materials available for this topic."

        subtopics = topic.subtopics if topic.subtopics else []
        query = f"{topic.title} {' '.join(subtopics)}"

        # Perform similarity search in the reference vector store
        results = self.reference_store.similarity_search(query, k=5)
        if not results:
            print(f"No reference materials found for topic: {topic.title}")
            return f"# {topic.title}\n\nNo reference materials available for this topic."

        reference_content = "\n\n".join([doc.page_content for doc in results])

        # Generate study material using the study material prompt
        chain = self.study_material_prompt | self.llm
        result = chain.invoke({
            "topic": topic.title,
            "description": topic.description,
            "reference_content": reference_content,
            "teacher_id": teacher_id
        })

        return result.content

    def generate_quiz_with_config(self, description: str, total_questions: int, duration: str, beginner: int, intermediate: int, advance: int) -> str:
        """
        Generate a quiz based on a description and quiz configuration.

        Args:
            description: Description of the quiz topic.
            total_questions: Number of total questions to generate.
            duration: Duration of the quiz in minutes.
            beginner: Number of beginner questions.
            intermediate: Number of intermediate questions.
            advance: Number of advanced questions.

        Returns:
            A JSON string representing the quiz.
        """
        if self.reference_store is None:
            print("No reference materials available. Please add reference materials first.")
            return json.dumps({"error": "No reference materials available"})

        # Perform similarity search in the reference vector store
        results = self.reference_store.similarity_search(description, k=2)
        reference_context = "\n\n".join([doc.page_content for doc in results]) if results else "No additional reference materials available."

        # Define the quiz prompt
        quiz_prompt = PromptTemplate(
            template="""Generate a quiz in JSON format based on the following configuration.

Quiz Description: {description}
Reference Materials: {reference_context}
Total Questions: {total_questions}
Question Levels Distribution: beginner {beginner}, intermediate {intermediate}, advanced {advance}
Duration: {duration} minutes

For each question, provide exactly four options labeled "a", "b", "c", and "d". The answer should be one of the four options: "a", "b", "c", or "d".

Output the result in valid JSON format with double quotes around all keys and values. The JSON format should be an array of question objects, as follows:

[
    {{
        "question": "Question text",
        "options": {{
            "a": "Option A text",
            "b": "Option B text",
            "c": "Option C text",
            "d": "Option D text"
        }},
        "answer": "Correct answer (a, b, c, or d)"
    }},
    ...
]

Please generate exactly {total_questions} questions distributed as follows:
Beginner: {beginner} questions,
Intermediate: {intermediate} questions,
Advanced: {advance} questions.
Do not include any additional commentary outside of the JSON.
""",
            input_variables=["description", "reference_context", "total_questions", "duration", "beginner", "intermediate", "advance"]
        )

        # Generate the quiz
        chain = quiz_prompt | self.llm
        result = chain.invoke({
            "description": description,
            "reference_context": reference_context,
            "total_questions": total_questions,
            "duration": duration,
            "beginner": beginner,
            "intermediate": intermediate,
            "advance": advance
        })

        return result.content

# Example usage
if __name__ == "__main__":
    study_system = LocalStudyMaterialRAG()

    # Add a syllabus from a PDF file
    syllabus_pdf_path = "cbse_physics12.pdf"  # Replace with the actual path to your syllabus PDF
    if os.path.exists(syllabus_pdf_path):
        syllabus_text = study_system.extract_text_from_pdf(syllabus_pdf_path)
        study_system.add_syllabus(syllabus_text, {"course_id": "CS101", "teacher_id": "T123"})
    else:
        print(f"Warning: Syllabus file {syllabus_pdf_path} not found")
        # Add some sample syllabus text for testing
        sample_syllabus = """
        Course: Introduction to Computer Science
        
        Module 1: Introduction to Programming
        - Basic programming concepts
        - Variables and data types
        - Control structures
        
        Module 2: Data Structures
        - Arrays and lists
        - Stacks and queues
        - Trees and graphs
        
        Module 3: Algorithms
        - Sorting algorithms
        - Searching algorithms
        - Complexity analysis
        """
        study_system.add_syllabus(sample_syllabus, {"course_id": "CS101", "teacher_id": "T123"})

    # Add reference materials from a PDF file
    reference_pdf_path = "atoms_class12.pdf"  # Replace with the actual path to your reference PDF
    if os.path.exists(reference_pdf_path):
        reference_text = study_system.extract_text_from_pdf(reference_pdf_path)
        study_system.add_reference_material(reference_text, {"source": "Physics", "teacher_id": "T123"})
    else:
        print(f"Warning: Reference file {reference_pdf_path} not found")
        # Add some sample reference material for testing
        sample_reference = """
        Data Structures and Algorithms
        
        Arrays are collections of elements identified by index or key. They are one of the most basic data structures.
        
        Lists are similar to arrays but can dynamically resize. In Python, lists are built-in and very versatile.
        
        Stacks follow the Last In First Out (LIFO) principle. Common operations include push (add an element) and pop (remove the most recently added element).
        
        Queues follow the First In First Out (FIFO) principle. Elements are added at the rear and removed from the front.
        
        Trees are hierarchical structures with a root value and subtrees of children with a parent node.
        
        Graphs consist of vertices (or nodes) connected by edges. They can be directed or undirected.
        
        Sorting algorithms arrange elements in a certain order. Common sorting algorithms include:
        - Bubble sort: O(n²) time complexity
        - Merge sort: O(n log n) time complexity
        - Quick sort: O(n log n) average time complexity
        
        Searching algorithms find an element in a data structure. Common searching algorithms include:
        - Linear search: O(n) time complexity
        - Binary search: O(log n) time complexity (requires sorted data)
        """
        study_system.add_reference_material(sample_reference, {"source": "Data Structures", "teacher_id": "T123"})

    # Generate materials for a single topic
    materials = study_system.create_full_course_materials("Data Structures", "T123")
    
    # Save the output to a .md file
    output_file = "study_materials.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write('''---
marp: true
theme: gaia
paginate: true
backgroundColor: "#1E1E2E"
color: white
\n''')
        for topic, content in materials.items():
            f.write('---\n')
            f.write(f"### {topic}\n\n")
            f.write(content)
            f.write("\n\n")

    print(f"Study materials saved to {output_file}")
    
    # Generate a quiz
    quiz_content = study_system.generate_quiz_with_config(
        description="Data Structures and Algorithms",
        total_questions=10,
        duration="10",
        beginner=3,
        intermediate=5,
        advance=2
    )
    
    quiz_file = "quiz.json"
    with open(quiz_file, "w", encoding="utf-8") as f:
        f.write(quiz_content)
    
    print(f"Quiz saved to {quiz_file}")
    
    # Try to convert to PPTX if marp is installed
    pptx_path = os.path.join('./', "slides.pptx")
    try:
        os.system(f"marp {output_file} --pptx -o {pptx_path}")
        print(f"Slides converted to PPTX: {pptx_path}")
    except Exception as e:
        print(f"Could not convert to PPTX: {e}")
        print("Please make sure Marp CLI is installed or run: npm install -g @marp-team/marp-cli")
