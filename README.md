
![Image](https://github.com/user-attachments/assets/9f655663-1c87-4149-9f64-48bfdcca00f7)

https://github.com/user-attachments/assets/5fa86ce2-d4bc-4613-b1fd-c083ec7b3c13



The utbots_llama package integrates Retrieval-Augmented Generation (RAG) with Meta's LLaMA language model within the Robot Operating System (ROS) framework. This enables robots to answer context-based questions by retrieving relevant information and generating coherent responses.
Features

    Contextual Question Answering: Enhances robot interactions by providing accurate, context-aware responses.
    Seamless ROS Integration: Ensures smooth communication between the RAG system and other ROS components.
    Modular Design: Facilitates easy customization and extension of functionalities.

Installation

    Clone the Repository:

git clone https://github.com/UtBotsAtHome-UTFPR/utbots_llama.git

Install Dependencies:

cd utbots_llama
pip install -r requirements.txt
Needs to perform: https://gist.github.com/defulmere/8b9695e415a44271061cc8e272f3c300 to work.

Build the Package:

    catkin_make

Usage

    Launch the RAG Node:

    roslaunch task_manager llama_qa.launch

    Interacting with the Robot: Use ROS topics or services to send questions and receive responses.

Configuration

    Model Settings: Adjust parameters in the config/model.yaml file to fine-tune the LLaMA model.
    Data Sources: Specify knowledge base locations in the config/data_sources.yaml file.

Contributing

We welcome contributions! Please fork the repository and submit a pull request with your changes.
License

This project is licensed under the MIT License.
