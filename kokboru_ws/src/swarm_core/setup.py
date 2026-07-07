from setuptools import setup
import os
from glob import glob

package_name = 'swarm_core'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ahmet',
    maintainer_email='ahmet@example.com',
    description='Decentralized Swarm Autonomy Package for Teknofest',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'swarm_agent = swarm_core.swarm_agent:main',
            'camera_node = swarm_core.camera_node:main',
            'uav_agent = swarm_core.uav_agent:main'
        ],
    },
)