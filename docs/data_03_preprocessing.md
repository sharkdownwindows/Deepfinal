# data-03 preprocessing notes

**subset selection**

The subsets were selected before viewing any training results. The selection order increases viewpoint coverage while keeping D1 inside D3, D3 inside D5, and D5 inside D10.

**cat_mug**

- D1: IMG_7898.jpeg
- D3: D1 + IMG_7901.jpeg + IMG_7904.jpeg
- D5: D3 + IMG_7899.jpeg + IMG_7902.jpeg
- D10: D5 + IMG_7900.jpeg + IMG_7903.jpeg + IMG_7905.jpeg + IMG_7916.jpeg + IMG_7917.jpeg

IMG_7898 is the front view. D3 adds IMG_7901 for the rear view and IMG_7904 for the top view. D5 adds IMG_7899 for a front-oblique view and IMG_7902 for a rear-oblique view. D10 adds the remaining side and intermediate views. IMG_7905 also provides a high-angle view from the right side with a different background.

**dog_plush**

- D1: IMG_7907.jpeg
- D3: D1 + IMG_7911.jpeg + IMG_7912.jpeg
- D5: D3 + IMG_7909.jpeg + IMG_7910.jpeg
- D10: D5 + IMG_7908.jpeg + IMG_7913.jpeg + IMG_7914.jpeg + IMG_7915.jpeg + IMG_7918.jpeg

IMG_7907 is the front view. D3 adds IMG_7911 for the rear view and IMG_7912 for the top view. D5 adds IMG_7909 and IMG_7910 to cover the left and right sides. D10 adds the remaining views. IMG_7913 provides a low-angle view that is still close to frontal, while IMG_7914 and IMG_7915 provide additional angled views with a different background.

**blue_white_vase**

- D1: IMG_7888.jpeg
- D3: D1 + IMG_7891.jpeg + IMG_7893.jpeg
- D5: D3 + IMG_7895.jpeg + IMG_7897.jpeg
- D10: D5 + IMG_7889.jpeg + IMG_7890.jpeg + IMG_7892.jpeg + IMG_7894.jpeg + IMG_7896.jpeg

IMG_7888 is used as the representative view. D3 adds two different rotations of the vase. D5 adds IMG_7895 for a higher-angle view that shows more of the opening and IMG_7897 for another rotation. D10 includes the remaining viewpoints.

**preprocessing**

All 30 training images were captured directly in a 1:1 format at 3024 x 3024, so no additional cropping is required. The preprocessing steps are:

- apply EXIF orientation
- convert to RGB
- resize from 3024 x 3024 to 512 x 512 using Lanczos interpolation
- no random horizontal flip

The preprocessing script checks that every source image is square before resizing. This prevents a non-square image from being resized and distorted accidentally.

**verification**

The preprocessing script checks the nested subset sizes and verifies that D1 is inside D3, D3 is inside D5, and D5 is inside D10. It also checks that each concept produces 10 processed images and that every processed image is 512 x 512 in RGB mode.
