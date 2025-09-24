#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String
from utbots_actions.action import InterpretLlama

from llama_cpp import Llama
import chromadb
# <<< IMPORTAÇÃO ADICIONADA PARA O ERRO CORRETO
from chromadb.errors import NotFoundError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download

import os
import time

REPO_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
FILENAME = "llama-2-7b-chat.Q2_K.gguf"

class LlamaActionServer(Node):
    def __init__(self):
        super().__init__('llama_action_server')

        self.pkg_path = get_package_share_directory('utbots_llm')
        
        self.get_logger().info("Inicializando ChromaDB...")
        self.client = chromadb.PersistentClient(path=self.pkg_path+"/.collections")
        # <<< LÓGICA CORRIGIDA: Criamos uma única coleção para todo o contexto
        self.collection = self.init_client()

        self.get_logger().info("Baixando e inicializando o modelo Llama...")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, cache_dir=self.pkg_path+"/.hf_cache")
        self.llm = Llama(
            model_path=model_path,
            verbose=False,
            n_ctx=2048,
            n_gpu_layers=-1                            
        )
        self.get_logger().info("Modelo Llama carregado.")

        self._action_server = ActionServer(
            self, InterpretLlama, 'llama_inference',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.sub_usermessage = self.create_subscription(String, '/utbots/voice/stt/whispered', self.callback_msg, 10)
        self.pub_response = self.create_publisher(String, '/utbots/voice/tts/robot_speech', 10)
        
        self.response = ""
        self.msg_whisper = String()
        self.new_msg = False
        
        self.get_logger().info("Servidor de Ação Llama pronto.")

    def handle_llama_service(self, req):
        input_prompt = """
        Answer the following question based on the context given after it in the same language as the question:
        ### Question:
        {}

        ### Context:
        {}

        ### Answer:
        {}"""
        questionQA=req
        # <<< LÓGICA CORRIGIDA: Buscamos na nossa coleção unificada
        search_context = self.collection.query(
            query_texts=[questionQA],
            n_results=3
        )
        context = search_context['documents'][0]
        input_text = input_prompt.format(questionQA, context, "")
        output = self.llm(input_text, max_tokens=100, stop=["###"], temperature=0.1, top_p=0.2, top_k=10, repeat_penalty=1.2, echo=True)
        response_text = output['choices'][0]['text']
        return (response_text)

    def callback_msg(self, msg):
        self.new_msg = True
        self.msg_whisper = msg
        self.get_logger().info(f'Recebida nova mensagem: "{msg.data}"')

    def goal_callback(self, goal_request):
        self.get_logger().info('Recebido novo goal.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Recebido pedido de cancelamento.')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executando o goal... Aguardando por uma requisição do usuário...')
        result = InterpretLlama.Result()

        while not self.new_msg:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled(); self.get_logger().info('Goal cancelado!'); return InterpretLlama.Result()
            time.sleep(0.5)

        try:
            whisper_msg = self.whisper_fix(self.msg_whisper.data)
            
            if not goal_handle.request.answer.data:
                if self.name_test(whisper_msg):
                    result.activation.data = True; self.response="Hi my name is Hestia. You can ask me a question"
                else:
                    result.activation.data = False; self.response="Hi my name is Hestia." 
            else:
                self.response = (self.handle_llama_service(whisper_msg)).split('### Answer:')[1]
                result.activation.data = False
            
            self.get_logger().info(f"[Llama] Request: {whisper_msg}")
            self.get_logger().info(f"[Llama] Response: {self.response}")

            result.llm_input.data = whisper_msg
            result.llm_output.data = self.response
            
            msg_response = String(); msg_response.data = self.response
            self.pub_response.publish(msg_response)
            self.get_logger().info("[Llama] Resposta publicada para o TTS.")

        except Exception as e:
            self.get_logger().error(f"[Llama] Erro: {e}"); goal_handle.abort(); return result

        self.new_msg = False
        goal_handle.succeed()
        self.get_logger().info("[Llama] Ação bem-sucedida. Enviando resultado.")
        return result

    def name_test(self, msg): return 'hestia' in msg.lower()
    def split_and_replace(self, string, word, new_word): return string.replace(word, new_word)
     
    def whisper_fix(self, msg):
        msg_lower = msg.lower()
        msg_lower = self.split_and_replace(msg_lower, 'robocop', 'robocup')
        msg_lower = self.split_and_replace(msg_lower, 'estro', 'hestia')
        msg_lower = self.split_and_replace(msg_lower, 'estia', 'hestia')
        return msg_lower
    
    # <<< LÓGICA CORRIGIDA E MELHORADA
    def init_client(self):
        collection_name = "utbots_context"
        try:
            # Tenta pegar a coleção. Se não existir, o 'except' será ativado.
            collection = self.client.get_collection(name=collection_name)
            self.get_logger().info(f"Coleção '{collection_name}' encontrada e carregada.")
            return collection
        except NotFoundError:
            # Se a coleção não existe, criamos ela e adicionamos os documentos.
            self.get_logger().info(f"Coleção '{collection_name}' não encontrada. Criando e populando...")
            collection = self.client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            
            PATH_DIR1 = self.pkg_path + "/resources/context/"
            splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50, separators=[".", " ", ""])
            
            all_chunks = []
            for file_name in os.listdir(PATH_DIR1):
                with open(os.path.join(PATH_DIR1, file_name), 'r') as f:
                    all_chunks.extend(splitter.split_text(f.read()))
            
            ids = [str(i) for i, _ in enumerate(all_chunks)]
            collection.add(documents=all_chunks, ids=ids)
            self.get_logger().info("Coleção criada e populada com sucesso.")
            return collection

def main(args=None):
    rclpy.init(args=args)
    llama_action_server = LlamaActionServer()
    try:
        rclpy.spin(llama_action_server)
    except KeyboardInterrupt: pass
    finally:
        llama_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()