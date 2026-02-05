import os
import cv2
import numpy as np

def detect_inconsistent_boundaries(mask):
    
    edges = cv2.Canny(mask.astype(np.uint8), 50, 150)

    total_edge_pixels = np.sum(edges > 0)
    if total_edge_pixels == 0:
        return 1.0, np.zeros_like(mask, dtype=np.float32)

    smoothed = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    diff = np.abs(mask.astype(np.float32) - smoothed)

    jagged_pixels = np.sum(diff > 0.5)
    confidence = 1.0 - (jagged_pixels / total_edge_pixels)

    return float(np.clip(confidence, 0, 1)), diff


def find_small_isolated_regions(mask, min_size=10):
   
    suspicious = []
    num_classes = int(mask.max()) + 1

    for cid in range(1, num_classes):  # skip background
        binary = (mask == cid).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_size:
                suspicious.append({
                    "class_id": int(cid),
                    "position": tuple(map(int, centroids[i])),
                    "size": int(area)
                })

    return suspicious


def detect_color_bleeding(image, mask, class_id):
    
    class_mask = (mask == class_id).astype(np.uint8) * 255
    mask_edges = cv2.Canny(class_mask, 100, 200)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_edges = cv2.Canny(gray, 50, 150)

    tolerance = cv2.dilate(image_edges, np.ones((3, 3), np.uint8))
    bleeding = cv2.bitwise_and(mask_edges, cv2.bitwise_not(tolerance))

    return bleeding


def calculate_annotation_confidence_map(mask):
    
    conf_map = np.ones(mask.shape, dtype=np.float32)
    _, jagged_map = detect_inconsistent_boundaries(mask)
    conf_map -= jagged_map * 0.4

    return np.clip(conf_map, 0, 1)


def compare_two_annotations(mask1, mask2):
   
    return float(np.mean(mask1 == mask2))


class AnnotationQualityReport:
    def __init__(self, image_folder, mask_folder, output_folder="quality_report"):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        self.visual_dir = os.path.join(output_folder, "visuals")

        os.makedirs(self.visual_dir, exist_ok=True)
        self.results = []

    def analyze_dataset(self):
        for file in os.listdir(self.image_folder):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(self.image_folder, file)
            mask_path = os.path.join(
                self.mask_folder, os.path.splitext(file)[0] + ".png"
            )

            if not os.path.exists(mask_path):
                continue

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            boundary_conf, jagged = detect_inconsistent_boundaries(mask)
            small_regions = find_small_isolated_regions(mask)

            conf_map = calculate_annotation_confidence_map(mask)


            heatmap = (conf_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(image, 0.65, heatmap, 0.35, 0)

            conf_path = os.path.join(self.visual_dir, f"{file}_conf.png")
            cv2.imwrite(conf_path, overlay)

            self.results.append({
                "file": file,
                "boundary_confidence": round(boundary_conf, 3),
                "small_regions": len(small_regions),
                "confidence_map": os.path.basename(conf_path)
            })

    def generate_html_report(self, output_file="quality_report.html"):
        html_path = os.path.join(self.output_folder, output_file)

        html = """
        <html>
        <head>
            <title>Annotation Quality Report</title>
            <style>
                body { font-family: Arial; margin: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; }
                .bad { background-color: #ffe6e6; }
            </style>
        </head>
        <body>
        <h1>Annotation Quality Report</h1>
        <table>
        <tr>
            <th>Image</th>
            <th>Boundary Confidence</th>
            <th>Small Regions</th>
            <th>Confidence Map</th>
        </tr>
        """

        for r in self.results:
            bad = "bad" if r["boundary_confidence"] < 0.7 or r["small_regions"] > 5 else ""
            html += f"""
            <tr class="{bad}">
                <td>{r['file']}</td>
                <td>{r['boundary_confidence']}</td>
                <td>{r['small_regions']}</td>
                <td><img src="visuals/{r['confidence_map']}" width="140"></td>
            </tr>
            """

        html += """
        </table>    
        </body>
        </html>
        """

        with open(html_path, "w") as f:
            f.write(html)

        print(f" HTML report generated → {html_path}")

if __name__ == "__main__":

    IMAGE_PATH = "sample_data/images/img1.jpg"
    MASK_PATH = "sample_data/masks/img1.png"

    image = cv2.imread(IMAGE_PATH)
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

    boundary_conf, jagged_map = detect_inconsistent_boundaries(mask)
    print("Boundary confidence:", boundary_conf)

    cv2.imwrite("debug_jagged_map.png", (jagged_map * 255).astype(np.uint8))

    small_regions = find_small_isolated_regions(mask, min_size=20)
    print("Small isolated regions:", small_regions)
    for cid in np.unique(mask):
        if cid == 0:
            continue
        bleeding = detect_color_bleeding(image, mask, int(cid))
        cv2.imwrite(f"bleeding_class_{cid}.png", bleeding)

    conf_map = calculate_annotation_confidence_map(mask)
    cv2.imwrite("confidence_map.png", (conf_map * 255).astype(np.uint8))

    agreement = compare_two_annotations(mask, mask)
    print("Annotation agreement:", agreement)

    IMAGE_DIR = "sample_data/images"
    MASK_DIR = "sample_data/masks"

    report = AnnotationQualityReport(
        image_folder=IMAGE_DIR,
        mask_folder=MASK_DIR
    )

    report.analyze_dataset()
    report.generate_html_report()
