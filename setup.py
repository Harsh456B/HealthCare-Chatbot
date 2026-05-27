"""
Setup configuration for Medical Chatbot package
"""

from setuptools import find_packages, setup

setup(
    name="medical_chatbot",
    version="0.1.0",
    description="A RAG-based medical chatbot using LangChain, Pinecone, and Groq",
    author="Medical Chatbot Team",
    packages=find_packages(),
    install_requires=[]
)
