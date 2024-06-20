import time
import cv2
import torch
import demo_utils
import noise_image
from thop import profile
import numpy as np
from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, viz2d
from lightglue.utils import load_image, rbd, numpy_image_to_torch
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    # The top image shows the matches, while the bottom image shows the point pruning across layers.
    # In this case, LightGlue prunes a few points with occlusions,
    # but is able to stop the context aggregation after 4/9 layers.

    torch.set_grad_enabled(False)
    img_path = Path("../assets")

    # -----------------------------------SuperPoint+LightGlue--------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)

    img0_path = "../assets/4SARSets/pair4-1.png"
    img1_path = "../assets/4SARSets/pair4-2.png"
    output_path = "../output/4SARSets/pair04.jpg"

    # --------------------Additive noise image ------------------
    # img0 = noise_image.Additive_noise(img0_path, 0)
    # output_path = "../output/1DSMsets/pair1+snr0.jpg"

    # --------------------stripe noise image --------------------
    # img0 = noise_image.stripe_noise(img0_path, 0.1)
    # output_path = "../output/1DSMsets/pair1+0p101S.jpg"

    img0 = cv2.imread(img0_path)
    img0 = demo_utils.resize(img0, 512)
    img1 = cv2.imread(img1_path)
    img1 = demo_utils.resize(img1, 512)

    image0 = numpy_image_to_torch(img0)
    image1 = numpy_image_to_torch(img1)

    # match the features
    with torch.no_grad():
        tic = time.time()
        # feats0 = extractor.extract(image0.to(device))
        # feats1 = extractor.extract(image1.to(device))
        feats0, img00 = extractor.extract(image0.to(device))
        feats1, img01 = extractor.extract(image1.to(device))

        matches01 = matcher({"image0": feats0, "image1": feats1})
        data = {"image0": feats0, "image1": feats1}
        toc = time.time()
        tt1 = toc - tic
        # print('run time:%.3f' % tt1)
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        corr0 = kpts0[matches[..., 0]].cpu().numpy()
        corr1 = kpts1[matches[..., 1]].cpu().numpy()

        # --------------------------RANSAC Outlier Removal----------------------------------
        # F_hat, mask_F = cv2.findFundamentalMat(corr0, corr1, method=cv2.USAC_ACCURATE,
        #                                        ransacReprojThreshold=1, confidence=0.99)
        F_hat, mask_F = cv2.findFundamentalMat(corr0, corr1, method=cv2.USAC_MAGSAC,
                                               ransacReprojThreshold=1, confidence=0.99)

        if mask_F is not None:
            mask_F = mask_F[:, 0].astype(bool)
        else:
            mask_F = np.zeros_like(corr0[:, 0]).astype(bool)

        # visualize match
        # display = demo_utils.draw_match(img0, img1, corr0, corr1)
        display = demo_utils.draw_match(img0, img1, corr0[mask_F], corr1[mask_F])

        putative_num = len(corr1)
        correct_num = len(corr1[mask_F])
        inliner_ratio = correct_num / putative_num
        # print(putative_num, correct_num)
        # print("inliner ratio:", inliner_ratio)

        text1 = "putative_num:{}".format(putative_num)
        text2 = 'correct_num:{}'.format(correct_num)
        text3 = 'inliner ratio:%.3f' % inliner_ratio
        text4 = 'run time: %.3fs' % tt1

        print('putative_num:{}'.format(putative_num), '\ncorrect_num:{}'.format(correct_num),
              '\ninliner ratio:%.3f' % inliner_ratio, '\nrun time: %.3fs' % tt1)

        cv2.putText(display, str(text1), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(display, str(text2), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(display, str(text3), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(display, str(text4), (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite(output_path, display)

        flops1, params1 = profile(extractor, inputs=({"image": img00},))
        flops2, params2 = profile(extractor, inputs=({"image": img01},))
        flops3, params3 = profile(matcher, inputs=(data,))

        print("Params1：", "%.2f" % (params1 / (1000 ** 2)), "M")
        print("GFLOPS1：", "%.2f" % (flops1 / (1000 ** 3)))
        print("Params2：", "%.2f" % (params2 / (1000 ** 2)), "M")
        print("GFLOPS2：", "%.2f" % (flops2 / (1000 ** 3)))
        print("Params3：", "%.2f" % (params3 / (1000 ** 2)), "M")
        print("GFLOPS3：", "%.2f" % (flops3 / (1000 ** 3)))
