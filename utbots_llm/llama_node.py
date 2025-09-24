import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import String
# O arquivo .action gera automaticamente as interfaces Goal, Result e Feedback
from utbots_actions.action import InterpretNLU

from llama_cpp import Llama
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download

import os
import time

# Constantes do modelo (sem alteração)
REPO_ID = "TheBloke/Llama-2-7B-Chat-GGUF"
FILENAME = "llama-2-7b-chat.Q2_K.gguf"

class LlamaActionServer(Node):
    def __init__(self):
        # 1. Inicialização do Nó
        super().__init__('llama_action_server')

        # 2. Obtenção do Caminho do Pacote
        self.pkg_path = get_package_share_directory('utbots_llm')
        
        # O restante da inicialização do Llama e ChromaDB permanece o mesmo
        self.get_logger().info("Inicializando ChromaDB...")
        self.client = chromadb.PersistentClient(path=self.pkg_path+"/.collections")
        self.init_client()

        self.get_logger().info("Baixando e inicializando o modelo Llama...")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, cache_dir=self.pkg_path+"/.hf_cache")
        self.llm = Llama(
            model_path=model_path,
            verbose=False,
            n_ctx=2048,
            n_gpu_layers=-1                            
        )
        self.get_logger().info("Modelo Llama carregado.")

        # 3. Inicialização do Action Server
        self._action_server = ActionServer(
            self,
            InterpretLlama,
            'llama_inference',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # 4. Publishers e Subscribers
        self.pub_response = self.create_publisher(String, '/utbots/voice/tts/robot_speech', 10)
        self.sub_usermessage = self.create_subscription(
            String, 
            '/utbots/voice/stt/whispered', 
            self.callback_msg, 
            10
        )
        
        # Variáveis de estado
        self.response = ""
        self.msg_whisper = String()
        self.new_msg = False
        
        self.get_logger().info("Servidor de Ação Llama pronto.")

    # Funções do Llama (handle_llama_service, whisper_fix, etc.) permanecem as mesmas
    # ... (O corpo das funções abaixo não precisa de alteração na sua lógica interna)
    def handle_llama_service(self, req):
        # ... (código inalterado)
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
            query_texts=[questionQA],
            n_results=3
        )
        context=search_context['documents'][0]
        input_text = input_prompt.format(
            questionQA,
            context,
            "",
        )
        output = self.llm( input_text, 
                        max_tokens=100, 
                        stop=["###"], 
                        temperature=0.1,
                        top_p=0.2,
                        top_k=10,
                        repeat_penalty=1.2,
                        echo=True
                        )
        response_text = output['choices'][0]['text']
        return (response_text)

    def name_test(self , msg):
        # ... (código inalterado)
        msg_n=msg.lower()
        if msg_n.find('hestia')==-1:
            return False
        return True
    
    def split_and_replace(self, string, word, new_word):
        # ... (código inalterado)
        split_str=string.split()
        new_string=''
        for element in split_str:
            if element==word:
                new_string=new_string+new_word+' '
            else:
                new_string=new_string+element+' '
        return new_string
 
    def whisper_fix(self, msg):
        # ... (código inalterado)
        msg=msg.lower()
        if msg.find('robocop')!=-1:
            msg=self.split_and_replace(msg,'robocop','robocup')
        if msg.find('estro')!=-1:
            msg=self.split_and_replace(msg,'estro','hestia')
        if msg.find('estia')!=-1:
            msg=self.split_and_replace(msg,'estia','hestia')
        return msg

    def init_client(self):
        # ... (código inalterado, exceto por um log)
        PATH_DIR1=self.pkg_path+"/resources/context/"
        # ... (o resto da função é igual)

    def callback_msg(self, msg):
        self.new_msg = True
        self.msg_whisper = msg
        self.get_logger().info(f'Recebida nova mensagem: "{msg.data}"')

    # 5. Callbacks do Action Server (ROS 2)
    def goal_callback(self, goal_request):
        """Aceita ou rejeita um novo goal."""
        self.get_logger().info('Recebido novo goal.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Aceita ou rejeita um pedido de cancelamento."""
        self.get_logger().info('Recebido pedido de cancelamento.')
        return CancelResponse.ACCEPT

    # 6. Lógica Principal da Ação (Execute Callback)
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executando o goal...')
        
        # Cria um objeto de resultado
        result = InterpretLlama.Result()

        self.get_logger().info("Aguardando por uma requisição do usuário...")
        while self.new_msg == False:
            # Verifica se o goal foi cancelado enquanto esperava
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal cancelado!')
                return InterpretLlama.Result()
            time.sleep(0.5) # Pequena pausa para não sobrecarregar o CPU

        try:
            whisper_msg = self.whisper_fix(self.msg_whisper.data)
            
            if goal_handle.request.Answer.data == False:
                if self.name_test(whisper_msg) == True:
                    result.Activation.data = True
                    self.response = "Hi my name is Hestia. You can ask me a question"
                else:
                    result.Activation.data = False
                    self.response = "Hi my name is Hestia." 
            else:
                self.response = (self.handle_llama_service(whisper_msg)).split('### Answer:')[1]
                result.Activation.data = False
            
            self.get_logger().info(f"[Llama] Request: {whisper_msg}")
            self.get_logger().info(f"[Llama] Response: {self.response}")

            # Preenche a mensagem de resultado
            result.LLMInput = String(data=whisper_msg)
            result.LLMOutput = String(data=self.response)
            
            # Publica a resposta para o TTS
            msg_response = String()
            msg_response.data = self.response
            self.pub_response.publish(msg_response)
            self.get_logger().info("[Llama] Resposta publicada para o TTS.")

        except Exception as e:
            self.get_logger().error(f"[Llama] Erro: {e}")
            goal_handle.abort()
            return result

        self.new_msg = False
        goal_handle.succeed()
        self.get_logger().info("[Llama] Ação bem-sucedida. Enviando resultado.")
        return result


# 7. Ponto de Entrada do Script (ROS 2)
def main(args=None):
    rclpy.init(args=args)
    llama_action_server = LlamaActionServer()
    try:
        rclpy.spin(llama_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        llama_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()