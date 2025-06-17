import chess
import chess.pgn

from src.mcts import TreeNode

iterations = 10000


def main():
    i = 0
    gn = chess.pgn.Game()
    mctn = TreeNode(1.414, gn)
    while not gn.board().is_game_over() and i < 20:
        for it in range(iterations):
            mctn.step()
            # if it % 100 == 0:
            #     print(i, it)

        mctn = mctn.get_best()
        mctn.parent = None
        # print(mctn.gamenode.move)
        gn.promote_to_main(mctn.gamenode.move)
        gn = gn.next()
        print(gn.san())
        # print(gn)
        i += 1

    # print(gn.game())
    print()
    for node in gn.game().mainline():
        if node.move is not None:
            print(node.san(), end=" ")


if __name__ == "__main__":
    main()
