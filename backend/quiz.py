# import random
# from transformers import pipeline

# generator = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base"
# )

# def generate_quiz(chunks, num_questions=5):
#     selected_chunks = random.sample(chunks, min(num_questions, len(chunks)))

#     quiz = []

#     for c in selected_chunks:
#         prompt = f"""
#         Create one multiple choice question from the text below.
#         Provide 1 correct answer and 3 wrong options.

#         Text:
#         {c['text']}

#         Format:
#         Question:
#         A)
#         B)
#         C)
#         D)
#         Correct:
#         """

#         result = generator(prompt, max_length=256)[0]["generated_text"]

#         lines = result.split("\n")
#         question = lines[0]
#         options = [l for l in lines if l.strip().startswith(("A", "B", "C", "D"))]
#         correct = next((l for l in lines if "Correct" in l), "")

#         quiz.append({
#             "question": question,
#             "options": options,
#             "correct": correct[-1] if correct else "A"
#         })

#     return quiz


