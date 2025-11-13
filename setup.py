from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'utbots_llm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'resources/context'), glob(os.path.join('resources/context', '*.*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ehg2004',
    maintainer_email='joaosiqueira.2005@alunos.utfpr.edu.br',
    description="The utbots_llm package integrates RAG with a Llama model in ROS 2.",
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'llama_server = utbots_llm.llama_node:main',
    ],
},
)