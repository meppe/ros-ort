
class Visualizer:
    def __init__(self):
        print("Initializing the Visualizer")

    def vis_detections(self, im, class_name, dets, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        print("Visualizing detection of class " + str(class_name))

        # switch red and blue
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                      thresh),
                     fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        # plt.draw()
        plt.savefig("kf_" + str(current_kf_id) + "_" + str(class_name) + ".png")
        print("image drawn")