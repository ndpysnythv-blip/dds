#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多设备实时同步服务器 - WebSocket
===============================
功能：
  1) 允许多个客户端连接
  2) 保存所有客户端共享的 localStorage 数据副本
  3) 任何客户端更新数据后，立即广播给所有其他客户端
  4) 新客户端首次连接时，从服务器拉取最新的全量数据

使用方法：
  1) 安装依赖： pip install websockets
  2) 运行：     python sync-server.py           # 默认端口 8765
                 python sync-server.py --port 9000 --host 0.0.0.0
  3) 查看本机 IP：Windows 用 ipconfig / Linux/Mac 用 ifconfig
     例如：本机 IP 为 192.168.1.100，端口 8765
     则在所有设备浏览器的 API 设置中填写： ws://192.168.1.100:8765

消息协议（JSON）：
  客户端 -> 服务器：
    {"type": "hello", "ts": ...}                          # 握手
    {"type": "pull",  "keys": ["key1","key2"]}           # 拉取最新数据
    {"type": "broadcast", "key": "...", "value": "..."}  # 广播某个 key 的变化

  服务器 -> 客户端：
    {"type": "broadcast", "key": "...", "value": "...", "ts": ...}    # 转发广播
    {"type": "pull-response", "data": { "key1": "value1", ... } }     # 全量数据响应
"""
import asyncio
import argparse
import json
import signal
import sys
from datetime import datetime

try:
    import websockets
except ImportError:
    print("[错误] 缺少依赖：websockets")
    print("请先执行：  pip install websockets")
    sys.exit(1)

# —— 全局状态 —— #
shared_data = {}                 # key -> value (字符串)
connected_clients = set()        # 当前连接的所有客户端

# 可选：日志颜色
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_RED = "\033[31m"
COLOR_RESET = "\033[0m"

def log(level, message):
    time_str = datetime.now().strftime("%H:%M:%S")
    color = {
        "INFO": COLOR_BLUE,
        "OK": COLOR_GREEN,
        "WARN": COLOR_YELLOW,
        "ERR": COLOR_RED,
    }.get(level, "")
    print(f"{color}[{time_str}] [{level}] {message}{COLOR_RESET}")


async def handle_client(websocket):
    """处理单个客户端连接的生命周期"""
    connected_clients.add(websocket)
    addr = websocket.remote_address
    client_id = f"{addr[0]}:{addr[1]}" if addr else "unknown"
    log("OK", f"客户端已连接： {client_id}  (当前在线 {len(connected_clients)} 人)")
    try:
        async for raw_msg in websocket:
            try:
                msg = json.loads(raw_msg)
            except Exception:
                continue
            msg_type = msg.get("type", "")

            if msg_type == "hello":
                # 新连接握手
                pass

            elif msg_type == "pull":
                # 客户端拉取最新全量数据
                keys = msg.get("keys", [])
                data = {k: shared_data.get(k, "") for k in keys}
                await websocket.send(json.dumps({"type": "pull-response", "data": data}))
                log("INFO", f"  ↳ 向 {client_id} 发送了全量数据 ({len(data)} 个key)")

            elif msg_type == "broadcast":
                # 客户端广播数据变化
                key = msg.get("key")
                value = msg.get("value")
                if key is None:
                    continue
                # 更新服务端副本
                shared_data[key] = value
                # 转发给所有其他连接中的客户端
                forward = json.dumps({
                    "type": "broadcast",
                    "key": key,
                    "value": value,
                    "ts": msg.get("ts", 0)
                })
                # 记录一下总共有多少条目
                log("OK", f"  ↳ 收到广播 key={key[:40]}{'...' if len(str(key)) > 40 else ''}  "
                         f"(数据总量 {len(shared_data)} 条)  转发给 {len(connected_clients)} 个客户端")

                # 异步转发（不等待每个客户端完成，避免阻塞）
                dead_clients = []
                for client in connected_clients:
                    if client.closed:
                        dead_clients.append(client)
                        continue
                    try:
                        await client.send(forward)
                    except Exception:
                        dead_clients.append(client)
                for dc in dead_clients:
                    connected_clients.discard(dc)

            else:
                # 未知消息类型，忽略
                pass

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        log("ERR", f"连接异常 {client_id}: {e}")
    finally:
        connected_clients.discard(websocket)
        log("WARN", f"客户端已断开：{client_id}  (当前在线 {len(connected_clients)} 人)")


async def main(host, port):
    log("INFO", "=" * 60)
    log("INFO", "多设备实时同步服务器 启动")
    log("INFO", f"监听地址:  ws://{host}:{port}")
    log("INFO", "按 Ctrl+C 停止服务器")
    log("INFO", "=" * 60)

    async with websockets.serve(handle_client, host, port, ping_interval=20, ping_timeout=60):
        await asyncio.Future()  # 永久运行


def signal_handler(sig, frame):
    print()
    log("WARN", "收到停止信号，正在关闭服务器...")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多设备实时同步服务器 (WebSocket)")
    parser.add_argument("--host", default="0.0.0.0", help="绑定的IP地址，默认 0.0.0.0 (所有网卡)")
    parser.add_argument("--port", type=int, default=8765, help="端口号，默认 8765")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        log("WARN", "服务器已停止")
    except Exception as e:
        log("ERR", f"启动失败: {e}")
        sys.exit(1)
