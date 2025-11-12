from src.geometry_optional.coords_classifier import Featurizer, CoordsClassifier
feat = Featurizer(grid_w=5, grid_h=3).featurize(boxes, img.width, img.height, los_y)
clf = CoordsClassifier()
clf.fit([feat1, feat2, ...], [label1, label2, ...])
y_hat = clf.predict(feat)
