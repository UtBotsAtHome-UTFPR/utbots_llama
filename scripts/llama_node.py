#!/usr/bin/env python3

import rospy
import rospkg   
from std_msgs.msg import String
from rospkg import RosPack
import actionlib
from utbots_actions.msg import InterpretLlamaAction, InterpretLlamaResult
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
FILENAME = "llama-2-7b-chat.Q2_K.gguf"

class LlamaInterface:
    def __init__(self):



        rospy.init_node('llama_action_node')
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('utbots_llama')

        #Initialize Chroma DB client
        self.client = chromadb.PersistentClient(path=self.pkg_path+"/.collections")
        self.init_client()

        # Initialize the Llama model
        model=hf_hub_download(repo_id=REPO_ID, filename=FILENAME,cache_dir=self.pkg_path+"/.hf_cache")
        self.llm = Llama(
            model_path=model,
            # embedding=True,
            verbose=False,
            # verbose=True,
            n_ctx=2048,
            n_gpu_layers = -1                            
                    )

        # Action server initialization
        self.server = actionlib.SimpleActionServer('llama_inference', InterpretLlamaAction, self.execute, False)
        # self.server.register_goal_callback(self.goalCB)
        self.server.start()

        # Publishers and Subscribers
        self.sub_usermessage = rospy.Subscriber('/utbots/voice/stt/whispered', String, self.callback_msg)
        self.pub_response = rospy.Publisher('/utbots/voice/tts/robot_speech', String, queue_size=1)
        #initalize variables
        self.response = ""
        self.enable_llama = False

        self.msg_whisper = String()
        self.new_msg = False
        

        self.rate = rospy.Rate(1)

        self.main()

    # def goalCB(self,goal):
    #     self.answer=goal.Answer.data
    #     self.new_goal=self.server.accept_new_goal()


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
        collection_name='all.txt'
        collection=self.client.get_collection(collection_name)
        search_context = collection.query(
        # query_embeddings=[emb_f(questionQA)],
        query_texts=[questionQA],
        # include=["documents"], 
        n_results=3
        )

        # print(search_context['documents'])

        context=search_context['documents'][0]

        input_text = input_prompt.format(
            questionQA,  # question
            # "",  # context
            context,  # context

            "",  # output - leave this blank for generation!
        )
        # print(questionQA)

        output = self.llm( input_text, 
                        max_tokens=100, 
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
        action_res = InterpretLlamaResult()

        rospy.loginfo(f"[Llama] Goal received, waiting for request")
        while self.new_msg == False:
            self.rate.sleep()
            if self.server.is_preempt_requested():
                rospy.loginfo("[Llama] Action preempted")
                self.server.set_preempted()
                return 
            
        try:
            whisper_msg=self.whisper_fix(self.msg_whisper.data)
            # print(goal.Answer)
            if goal.Answer.data == False:
            # if False:
                # if True:
                if self.name_test(whisper_msg)==True:
                    action_res.Activation.data = True
                    self.response="Hi my name is Hestia. You can ask me a question"
                else:
                    action_res.Activation.data = False
                    self.response="Hi my name is Hestia." 
            else:
                self.response = (self.handle_llama_service(whisper_msg)).split('### Answer:')[1] # llama response
                action_res.Activation.data = False


            rospy.loginfo(f"[Llama] Request: {whisper_msg}")
            rospy.loginfo(f"[Llama] Response: {self.response}")
            action_res.LLMInput = String(whisper_msg)
            action_res.LLMOutput = String(self.response)

            rospy.loginfo("[Llama] TTS response published")

            self.msg_response = String()
            self.msg_response.data = self.response
            # print(self.response[])
            self.pub_response.publish(self.msg_response)
                
        except Exception as e:
            rospy.logwarn(f"[Llama] Error: {e}")
            self.server.set_aborted()

        self.new_msg = False
        rospy.loginfo("[Llama] Action succeded. Sending result")
        self.server.set_succeeded(action_res)

    def name_test(self , msg):
        msg_n=msg.lower()
        # print(msg_n.find('hestia'))
        if msg_n.find('hestia')==-1:
            return False #to be
        return True
    
    def split_and_replace(self, string, word, new_word):
        split_str=string.split()
        new_string=''
        for element in split_str:
            if element==word:
                new_string=new_string+new_word+' '
            else:
                new_string=new_string+element+' '
        return new_string

 
    def whisper_fix(self, msg):
        msg=msg.lower()
        if msg.find('robocop')!=-1:
            msg=self.split_and_replace(msg,'robocop','robocup')
        if msg.find('estro')!=-1:
            msg=self.split_and_replace(msg,'estro','hestia')
        if msg.find('estia')!=-1:
            msg=self.split_and_replace(msg,'estia','hestia')
        return msg
    


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
            # print(file_name)
            new_file=open((PATH_DIR1+file_name),'r')
            # collection_name=file_name.removesuffix(".txt")
            collection_name=file_name
            try:
                self.client.get_collection(name=collection_name)
            except chromadb.errors.InvalidCollectionException:
                # if collection_name=="RoboCup.txt":
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