import streamlit as st
import os
import json
import uuid
from LangChainModel import LocalStudyMaterialRAG  # Assuming LangChainModel.py is in the same directory

# Initialize the study system
study_system = LocalStudyMaterialRAG()

# Define the path for saving generated quiz and PPT
output_dir = "generated_files"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Title of the application
st.title("Quiz and PPT Generator")

# File upload widgets
syllabus_pdf = st.file_uploader("Upload Syllabus PDF", type="pdf")
reference_pdf = st.file_uploader("Upload Reference PDF", type="pdf")

# Text inputs for additional configurations
total_questions = st.number_input("Total Questions", min_value=1, value=10)
duration = st.text_input("Duration (in minutes)", value="30")
beginner = st.number_input("Beginner Questions", min_value=0, value=3)
intermediate = st.number_input("Intermediate Questions", min_value=0, value=4)
advance = st.number_input("Advanced Questions", min_value=0, value=3)

# Button to generate quiz and PPT
if st.button("Generate Quiz and PPT"):
    if syllabus_pdf and reference_pdf:
        # Save uploaded files locally
        syllabus_path = os.path.join(output_dir, f"syllabus_{uuid.uuid4().hex}.pdf")
        reference_path = os.path.join(output_dir, f"reference_{uuid.uuid4().hex}.pdf")

        with open(syllabus_path, "wb") as f:
            f.write(syllabus_pdf.read())

        with open(reference_path, "wb") as f:
            f.write(reference_pdf.read())

        # Process the syllabus and reference files to generate quiz and PPT
        syllabus_text = study_system.extract_text_from_pdf(syllabus_path)
        study_system.process_syllabus(syllabus_text)

        reference_text = study_system.extract_text_from_pdf(reference_path)
        study_system.add_reference_material(reference_text, {"source": "Reference Material"})

        # Extract topics from the syllabus
        topics = study_system.extract_topics("Generated Topic")
        if not topics:
            st.error("No topics found in the syllabus.")
            st.stop()

        # Use the first topic for generating study material
        topic = topics[0]

        # Generate the quiz and PPT
        quiz_content = study_system.generate_quiz_with_config(
            description="Generated Quiz",  # You can modify this as needed
            total_questions=total_questions,
            duration=duration,
            beginner=beginner,
            intermediate=intermediate,
            advance=advance
        )

        ppt_content = study_system.generate_study_material(
            topic=topic,  # Pass the Topic object
            teacher_id="Generated Teacher"
        )

        # Save the generated quiz to a file
        quiz_filename = f"quiz_{uuid.uuid4().hex}.json"
        quiz_path = os.path.join(output_dir, quiz_filename)

        with open(quiz_path, "w", encoding="utf-8") as f:
            f.write(quiz_content)

        # Save the generated PPT to a file
        ppt_filename = f"ppt_{uuid.uuid4().hex}.md"
        ppt_path = os.path.join(output_dir, ppt_filename)

        with open(ppt_path, "w", encoding="utf-8") as f:
            f.write(ppt_content)

        # Provide download links for the generated files
        st.success("Quiz and PPT generated successfully!")

        # Download links
        st.download_button(
            label="Download Quiz",
            data=open(quiz_path, "rb").read(),
            file_name=quiz_filename,
            mime="application/json"
        )

        st.download_button(
            label="Download PPT",
            data=open(ppt_path, "rb").read(),
            file_name=ppt_filename,
            mime="application/markdown"
        )
    else:
        st.error("Please upload both the syllabus and reference PDFs.")
