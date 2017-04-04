import os
import json
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def sort_bbox(obj):
    return (-1.0)*obj['score']

def load_results(result_files):
    images = {}
    for result_file in result_files:
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = line.split()
                    image_name = row[0]
                    score = float(row[1])
                    box = [float(c) for c in row[2:6]]
                    num_recog = row[6]
                    if image_name not in images:
                        images[image_name] = {}
                    if result_file not in images[image_name]:
                        images[image_name][result_file] = []
                    images[image_name][result_file].append(
                      {'score': score, 'box': box, 'num_recog': num_recog}
                    )
    for image_name in images:
        for result_file in images[image_name]:
            images[image_name][result_file].sort(key=sort_bbox)
    return images
    
def vis_detections(num_cols, all_bboxes, image_folder, 
        image_extension='jpg', always_show_highest=True, 
        thresh=0.05, filename_mapping=None):
    """Visual debugging of detections."""
    if filename_mapping:
        with open(filename_mapping) as f:
            mapping = json.load(f)['testing']
    colors = ['r', 'g', 'b', 'c']
    for image_name, bboxes_n_scores in all_bboxes.items():
        im_path = os.path.join(image_folder, image_name+('.'+image_extension if image_extension else ''))
        im = cv2.imread(im_path)
        im = im[:, :, (2, 1, 0)]

        num_rows = int(max(len(bboxes_n_scores)/num_cols, 1))
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, 
                sharex=True, sharey=True)
        row_idx = 0
        col_idx = 0
        for result_file in bboxes_n_scores:
            if num_rows > 1:
                ax = axs[row_idx][col_idx]
            else:
                if num_cols > 1:
                    ax = axs[col_idx]
                else:
                    ax = axs
            ax.cla()
            img = ax.imshow(im)
            for i in range(len(bboxes_n_scores[result_file])):
                bbox = bboxes_n_scores[result_file][i]['box']
                score = bboxes_n_scores[result_file][i]['score']
                num_recog = bboxes_n_scores[result_file][i]['num_recog']
                if score > thresh or (i==0 and always_show_highest):
                    ax.add_patch(
                        patches.Rectangle((bbox[0], bbox[1]),
                                      bbox[2] - bbox[0],
                                      bbox[3] - bbox[1], fill=False,
                                      edgecolor=colors[row_idx*num_cols+col_idx], linewidth=1.5)
                        )
                    ax.annotate('%s' % (num_recog), xy=[bbox[0], bbox[1]], 
                      fontsize=12, fontweight='bold', color='r')
            ax.set_title(os.path.basename(result_file))
            if filename_mapping:
                ax.set_xlabel(os.path.basename(mapping[image_name]))
            col_idx += 1
            if col_idx >= num_cols:
                row_idx += 1
                col_idx = 0
        
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Show the detection results using Matplotlib')
    parser.add_argument('--result_files', dest='result_files', 
            help='The detection result files generated by py-faster-rcnn, could be multiple.',
            required=True, nargs='+')
    parser.add_argument('--image_folder', dest='image_folder',
            help='The image folder',
            default='', type=str)
    parser.add_argument('--img_ext', dest='image_extension',
            help='The image extension',
            default='jpg', type=str)
    parser.add_argument('--always_show_highest', action='store_true',
            help='whether to always show the bbox with highest score, even it belows the threshold')
    parser.add_argument('--thres', dest='threshold',
            help='The bbox score threshold. Scores below the threshold will not be shown. Range [0, 1]',
            default=0.05, type=float)

    args = parser.parse_args()
    
    all_bboxes = load_results(args.result_files)
    vis_detections(
        len(args.result_files), 
        all_bboxes, 
        args.image_folder,
        args.image_extension,
        args.always_show_highest,
        args.threshold
    )
