"""Minimal UCI engine used by harness tests: plays the first legal move."""

import sys

import chess

board = chess.Board()
OPTIONS = {
    "UCI_LimitStrength": ("check", "false"),
    "UCI_Elo": ("spin", "2000"),
    "Threads": ("spin", "1"),
    "Hash": ("spin", "16"),
    "Nodes": ("spin", "10000"),
}


def send(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def set_position(parts):
    global board
    if "startpos" in parts:
        board = chess.Board()
    elif "fen" in parts:
        start = parts.index("fen") + 1
        end = parts.index("moves") if "moves" in parts else len(parts)
        board = chess.Board(" ".join(parts[start:end]))
    if "moves" in parts:
        for token in parts[parts.index("moves") + 1 :]:
            board.push_uci(token)


for raw in sys.stdin:
    parts = raw.split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "uci":
        send("id name stub")
        send("id author tests")
        for name, (kind, default) in OPTIONS.items():
            if kind == "check":
                send(f"option name {name} type check default {default}")
            else:
                send(f"option name {name} type spin default {default} min 0 max 3000000")
        send("uciok")
    elif cmd == "isready":
        send("readyok")
    elif cmd == "ucinewgame":
        board = chess.Board()
    elif cmd == "position":
        set_position(parts)
    elif cmd == "go":
        move = next(iter(board.legal_moves), None)
        send(f"bestmove {move.uci() if move else '0000'}")
    elif cmd == "quit":
        break
