#!/usr/bin/env python3

import rospy
import rospkg   
from std_msgs.msg import String
from rospkg import RosPack
import actionlib
from utbots_actions.msg import InterpretNLUAction, InterpretNLUResult
from std_msgs.msg import String
from llama_cpp import Llama
import chromadb
# from chromadb.utils import embedding_functions
# emb_f = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
from langchain.text_splitter import RecursiveCharacterTextSplitter
#pip install -qU langchain-text-splitters
from typing import List
import numpy as np
from timeit import default_timer as timer
import os
from itertools import islice
import pandas as pd
from huggingface_hub import hf_hub_download
REPO_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
FILENAME = "TheBloke/llama-2-7b-chat.Q2_K.gguf"

class LlamaInterface:
    def __init__(self):



        rospy.init_node('llama_action_node')
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('utbots_llama_node')
        #Initialize Chroma DB client
        self.client = chromadb.PersistentClient(path=self.pkg_path+"/.collections")

        self.init_client()
        # Initialize the Llama model
        self.llm = Llama.from_pretrained(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            filename="llama-2-7b-chat.Q2_K.gguf",
            # embedding=True,
            verbose=True,
            n_ctx=4096,
            # verbose=False,
            n_gpu_layers = -1                            
                        )
        

        # Action server initialization
        self.server = actionlib.SimpleActionServer('llama_inference', InterpretNLUAction, self.execute, False)
        self.server.start()

        # Publishers and Subscribers
        self.sub_usermessage = rospy.Subscriber('/utbots/voice/stt/whispered', String, self.callback_msg)
        self.pub_response = rospy.Publisher('/utbots/voice/tts/robot_speech', String, queue_size=1)

        self.response = ""
        self.enable_llama = False

        self.msg_whisper = String()
        self.new_msg = False

        self.rate = rospy.Rate(1)

        self.main()

    def handle_llama_service(self, req):
        # Process the request using the Llama model
        input_prompt = """
        Answer the following question based on the context given after it in the same language as the question:
        ### Question:
        {}

        ### Context:
        {}

        ### Answer:
        {}"""
        questionQA=req
        collection_name='RoboCup.txt'
        collection=self.client.get_collection(collection_name)
        search_context = collection.query(
        # query_embeddings=[emb_f(questionQA)],
        query_texts=[questionQA],
        # include=["documents"], 
        n_results=4
        )

        # print(search_context['documents'])

        search_context['documents']
        context=search_context['documents'][0]

        input_text = input_prompt.format(
            questionQA,  # question
            # "",  # context
            context,  # context

            "",  # output - leave this blank for generation!
        )
        print(questionQA)

        output = self.llm( input_text, 
                        max_tokens=150, 
                        stop=["###"], 
                        temperature=0.1,
                        top_p=0.2,
                        top_k=10,
                        repeat_penalty=1.2,
                        echo=True
                        )

        
        # Prepare and return the response
        response_text = output['choices'][0]['text']
        return (response_text)

    def callback_msg(self, msg):
        self.new_msg = True
        self.msg_whisper = msg

    def execute(self, goal):
        rospy.loginfo(f"[Llama] Goal received, waiting for request")
        while self.new_msg == False:
            self.rate.sleep()
            if self.server.is_preempt_requested():
                rospy.loginfo("[Llama] Action preempted")
                self.server.set_preempted()
                return 
        try:
            self.response = self.handle_llama_service(self.msg_whisper.data) # llama response
            rospy.loginfo(f"[Llama] Request: {self.msg_whisper.data}")
            rospy.loginfo(f"[Llama] Response: {self.response}")
            action_res = InterpretNLUResult()
            action_res.NLUInput = String(self.msg_whisper.data)
            action_res.NLUOutput = String(self.response)
        
            rospy.loginfo("[Llama] TTS response published")

            self.msg_response = String()
            self.msg_response.data = self.response
            self.pub_response.publish(self.msg_response)

        except Exception as e:
            rospy.logwarn(f"[Llama] Error: {e}")
            self.server.set_aborted()

        self.new_msg = False
        rospy.loginfo("[Llama] Action succeded. Sending result")
        self.server.set_succeeded(action_res)

    def init_client(self):
        PATH_DIR1=self.pkg_path+"/resources/context/"

        splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        separators=[".", " ", ""]
        )

        name_of_files = os.listdir(PATH_DIR1)

        files = []
        for file_name in name_of_files:
            #files.append(new_file.read())
            print(file_name)
            new_file=open((PATH_DIR1+file_name),'r')
            # collection_name=file_name.removesuffix(".txt")
            collection_name=file_name
            try:
                self.client.get_collection(name=collection_name)
            except chromadb.errors.InvalidCollectionException:
                if collection_name=="RoboCup.txt":
                    #print("Collection not found. Appending...")
                    chunks = splitter.split_text(new_file.read())
                    file_list = chunks
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"} # l2 is the default
                    )
                    file_list_removed = []
                    embeddings=[]
                    ids=[]
                    for i in range(0,len(chunks)):
                        ids.append(str(i))
                    collection.add(
                        documents=chunks,
                        # embeddings=embeddings,
                        # metadatas=[]
                        ids=ids
                    ) 

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == "__main__":
    LlamaInterface()