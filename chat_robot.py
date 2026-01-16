#!/usr/bin/env python3
"""
대화형 로봇 명령 인터페이스
Interactive Robot Command Interface
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ChatInterface(Node):
    def __init__(self):
        super().__init__('chat_interface')
        self.publisher = self.create_publisher(String, '/llm_command', 10)
        self.status_sub = self.create_subscription(
            String, '/delivery_status', self.status_callback, 10)
        self.last_status = ""

    def status_callback(self, msg):
        if msg.data != self.last_status:
            self.last_status = msg.data
            print(f"\n🤖 로봇: {msg.data}")
            print("You: ", end="", flush=True)

    def send_command(self, command):
        msg = String()
        msg.data = command
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = ChatInterface()

    print("=" * 50)
    print("🤖 로봇 대화형 인터페이스")
    print("=" * 50)
    print("사용 가능한 명령어 예시:")
    print("  - 'pick station으로 가줘'")
    print("  - '원점으로 이동해'")
    print("  - 'drop station 가줘'")
    print("  - '좌표 (2, 3)으로 이동해'")
    print("종료: 'quit' 또는 'exit'")
    print("=" * 50)

    import threading
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q', '종료']:
                print("👋 종료합니다!")
                break
            if user_input:
                node.send_command(user_input)
                print("📤 명령 전송됨...")
    except KeyboardInterrupt:
        print("\n👋 종료합니다!")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
