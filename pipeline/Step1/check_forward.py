from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("forward_paddle.npy")
    paddle_info = diff_helper.load_info("forward_torch.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(diff_method="mean", path="forward_diff.log")
