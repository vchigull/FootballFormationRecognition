import argparse, torch
from torchvision.models import resnet50
from PIL import Image
import json
from transforms import make_eval_transform

def load_model(ckpt_path):
    model = resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # placeholder changed after load
    ckpt = torch.load(ckpt_path, map_location='cpu')
    classes = ckpt['classes']
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, classes

@torch.no_grad()
def predict(model, t, img_path, class_names):
    img = Image.open(img_path).convert('RGB')
    x = t(img).unsqueeze(0)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    conf, idx = prob.max(dim=0)
    return class_names[idx], float(conf)

def main(a):
    t = make_eval_transform(a.img_size)
    off_model, off_classes = load_model(a.offense_ckpt)
    def_model, def_classes = load_model(a.defense_ckpt)
    off_label, off_conf = predict(off_model, t, a.image, off_classes)
    def_label, def_conf = predict(def_model, t, a.image, def_classes)
    print(json.dumps({
        "image": a.image,
        "offense": {"label": off_label, "confidence": off_conf, "classes": off_classes},
        "defense": {"label": def_label, "confidence": def_conf, "classes": def_classes},
    }, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--offense_ckpt', default='checkpoints/best_offense.pt')
    ap.add_argument('--defense_ckpt', default='checkpoints/best_defense.pt')
    ap.add_argument('--img_size', type=int, default=640)
    main(ap.parse_args())
