import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String
from utbots_actions.action import InterpretLlama # A importação continua a mesma

from llama_cpp import Llama
import chromadb
from chromadb.errors import NotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download

import os


class LlamaActionServer(Node):
    def __init__(self):
        super().__init__('llama_action_server')

        self.pkg_path = get_package_share_directory('utbots_llm')
        
        self.get_logger().info("Inicializando ChromaDB...")
        self.client = chromadb.PersistentClient(path=self.pkg_path+"/.collections")
        self.collection = self.init_client()

        self.get_logger().info("Configurando e inicializando o modelo Llama...")
        
        self.get_logger().info("Configurando e inicializando o modelo Llama...")
        
        # Define o caminho fixo direto para o modelo Llama 3.1 8B local
        model_path = "/home/joao/ros_joao/src/utbots_llama/resources/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"        
       
        self.llm = Llama(
            model_path=model_path,
            verbose=False,
            n_ctx=2048,
            n_gpu_layers= 0                            
        )

        self._action_server = ActionServer(
            self,
            InterpretLlama,
            'llama_inference',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.pub_response = self.create_publisher(String, '/utbots/voice/tts/robot_speech', 10)
   
        
        self.get_logger().info("Servidor de Ação Llama pronto para receber goals.")

    def handle_llama_service(self, req):
        input_prompt = """
        You are a helpful and direct robot assistant. Answer the following question in a natural, conversational tone based on the provided data.

        CRITICAL REASONING RULES:
        1. Answer directly and concisely. Do NOT include any markdown formatting, hashes (#), list numbers (like '1.'), or introductory phrases.
        2. Do NOT mention the words "context", "text", "document", "provided data", or "according to the text". Act as if you know the information yourself.
        3. NEVER use digits or numbers in your response. Every single number, score, date, year, or quantity MUST be spelled out in Portuguese text.

        EXAMPLES OF NUMBER CONVERSIONS:
        - 4 -> quatro
        - 48 -> quarenta e oito
        - 3 a 1 -> três a um
        - 2026 -> dois mil e vinte e seis
        - 18 anos -> dezoito anos
        - 88.966 -> oitenta e oito mil novecentos e sessenta e seis

        ### Question:
        {}

        ### Context:
        {}

        ### Answer:
        {}"""
        questionQA=req
        search_context = self.collection.query(query_texts=[questionQA], n_results=3)
        context = search_context['documents'][0]
        input_text = input_prompt.format(questionQA, context, "")
        output = self.llm(
            input_text, 
            max_tokens=50, 
            stop=["###", "\n\n", "CRITICAL"], 
            temperature=0.1, 
            top_p=0.2, 
            top_k=10, 
            repeat_penalty=1.2, 
            echo=False
        )
        response_text = output['choices'][0]['text']
        return (response_text)


    def goal_callback(self, goal_request):
        self.get_logger().info('Recebido novo goal.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Recebido pedido de cancelamento.')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executando o goal...')
 
        result = InterpretLlama.Result()

        # Agora pegamos o texto diretamente do "goal".
        request_text = goal_handle.request.text_input.data

        
        try:
            # Limpa e processa o texto recebido do Whisper
            clean_text = self.whisper_fix(request_text)
            
            # Pega a resposta direta vinda do serviço
            raw_response = self.handle_llama_service(clean_text)
            
            # Se por acaso o marcador ainda vier junto, limpamos aqui
            if '### Answer:' in raw_response:
                response_text = raw_response.split('### Answer:')[1]
            else:
                response_text = raw_response
            
            # Limpeza preventiva de qualquer vazamento de regras
            if "CRITICAL" in response_text:
                response_text = response_text.split("CRITICAL")[0]
                
            response_text = response_text.strip()
            
            self.get_logger().info(f"[Llama] Request: {clean_text}")
            self.get_logger().info(f"[Llama] Response limpa: {response_text}")

            # Preenchemos o novo campo de resultado 'llm_output'.
            result.llm_output.data = response_text

        except Exception as e:
            self.get_logger().error(f"[Llama] Erro: {e}")
            goal_handle.abort()
            return result

        goal_handle.succeed()
        self.get_logger().info("[Llama] Ação bem-sucedida. Enviando resultado.")
        return result

    def whisper_fix(self, msg):
        msg_lower = msg.lower()
        msg_lower = msg_lower.replace('robocop', 'robocup')
        msg_lower = msg_lower.replace('estro', 'hestia')
        msg_lower = msg_lower.replace('estia', 'hestia')
        return msg_lower
    
    def init_client(self):
        collection_name = "utbots_context"
        try:
            collection = self.client.get_collection(name=collection_name)
            self.get_logger().info(f"Coleção '{collection_name}' encontrada e carregada.")
            return collection
        except NotFoundError:
            self.get_logger().info(f"Coleção '{collection_name}' não encontrada. Criando e populando...")
            collection = self.client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            PATH_DIR1 = "/home/joao/ros_joao/src/utbots_llama/resources/context/"   
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50, separators=["\n\n", "\n", ".", " ", ""])
            all_chunks = []
            for file_name in os.listdir(PATH_DIR1):
                with open(os.path.join(PATH_DIR1, file_name), 'r', encoding='utf-8') as f:
                    all_chunks.extend(splitter.split_text(f.read()))
            ids = [str(i) for i, _ in enumerate(all_chunks)]
            collection.add(documents=all_chunks, ids=ids)
            self.get_logger().info("Coleção criada e populada com sucesso.")
            return collection

def main(args=None):
    rclpy.init(args=args)
    llama_action_server = LlamaActionServer()
    try: rclpy.spin(llama_action_server)
    except KeyboardInterrupt: pass
    finally:
        llama_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
