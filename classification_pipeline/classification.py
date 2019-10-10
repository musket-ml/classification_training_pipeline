from musket_core import configloader, datasets, generic_config as generic

import keras.applications as apps
import os
import keras
import tqdm
import numpy as np
import classification_models
from segmentation_models import backbones

def create_back_bone(name):
    def create(*args, **kwargs):
        return backbones.get_backbone(name.lower(), *args, **kwargs)

    return create

extra_train=generic.extra_train
import tensorflow as tf
class ClassificationPipeline(generic.GenericImageTaskConfig):

    def __init__(self,**atrs):
        super().__init__(**atrs)
        self.dataset_clazz = datasets.KFoldedDataSet4ImageClassification
        self.flipPred=False

        pass

    def createStage(self,x):
        return ClassificationStage(x,self)

    def update(self, z, res):
        z.results = res
        pass

    def createNet(self):
        if self.architecture in configloader.load("layers").catalog:
            clazz=configloader.load("layers").catalog[self.architecture].func
        elif self.architecture.lower() in backbones.get_names():
            clazz = create_back_bone(self.architecture)
        else:    
            try:
                clazz = classification_models.Classifiers.get_classifier(self.architecture.lower())
            except:        
                try:
                     clazz = getattr(apps, self.architecture)
                except:  
                    print("Unknown architeture:"+self.backbone)
                    print ("Known archectures:",classification_models.Classifiers.names());
                    raise ValueError("Unknown architecture")
        t: configloader.Type = configloader.loaded['classification'].catalog['ClassificationPipeline']
        r = t.customProperties()
        cleaned = {}
        for arg in self.all:
            pynama = t.alias(arg)
            if not arg in r:
                cleaned[pynama] = self.all[arg]

        self.clean(cleaned)

        if self.crops is not None:
            cleaned["input_shape"]=(cleaned["input_shape"][0]//self.crops,cleaned["input_shape"][1]//self.crops,cleaned["input_shape"][2])
            
        cleaned["input_shape"]=tuple(cleaned["input_shape"])
        cleaned["include_top"]=False
        model1= self.__inner_create(clazz, cleaned)
        cuout=model1.output
        if len(cuout.shape) == 4:
            cuout=keras.layers.GlobalAveragePooling2D()(cuout)
        ac=self.all["activation"];
        if ac=="none":
            ac=None
        if self.dropout>0:
            cuout=keras.layers.Dropout(self.dropout)(cuout)
        if isinstance(ac, list):
           ls=[]
           for i in range(len(ac)):
               dl = keras.layers.Dense(self.all["classes"][i], activation=ac[i])(cuout)
               ls.append(dl)
           model = keras.Model(model1.input,ls)    
        else:        
            dl = keras.layers.Dense(self.all["classes"], activation=ac)(cuout)
            model = keras.Model(model1.input, dl)
        return model

    def predict_in_directory(self, spath, fold, stage, cb, data, limit=-1, batch_size=32, ttflips=False):
        with tqdm.tqdm(total=len(generic.dir_list(spath)), unit="files", desc="classification of images from " + str(spath)) as pbar:
            for v in self.predict_on_directory(spath, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b:imgaug.Batch=v
                for i in range(len(b.data)):
                    id=b.data[i]
                    cb(id,id,data)
                pbar.update(batch_size)

    def evaluateAll(self,ds, fold:int,stage=-1,negatives="real",ttflips=None,batchSize=32):
        folds = self.kfold(ds, range(0, len(ds)),batch=batchSize)
        vl, vg, test_g = folds.generator(fold, False,negatives=negatives,returnBatch=True)
        indexes = folds.sampledIndexes(fold, False, negatives)
        m = self.load_model(fold, stage)
        num=0
        with tqdm.tqdm(total=len(indexes), unit="files", desc="segmentation of validation set from " + str(fold)) as pbar:
            try:
                for f in test_g():
                    if num>=len(indexes): break
                    x, y, b = f
                    z = self.predict_on_batch(m,ttflips,b)
                    ids=b.data[0]
                    b.results=z;
                    b.ground_truth=b.data[1]
                    yield b
                    num=num+len(z)
                    pbar.update(len(ids))
            finally:
                vl.terminate()
                vg.terminate()
        pass

    def evaluate_all_to_arrays(self,ds, fold:int,stage=-1,negatives="real",ttflips=None,batchSize=32):
        lastFullValPred = None
        lastFullValLabels = None
        for v in self.evaluateAll(ds, fold, stage,negatives,ttflips,batchSize):
            if lastFullValPred is None:
                lastFullValPred = v.results
                lastFullValLabels = v.ground_truth
            else:
                lastFullValPred = np.append(lastFullValPred, v.results, axis=0)
                lastFullValLabels = np.append(lastFullValLabels, v.ground_truth, axis=0)
        return lastFullValPred,lastFullValLabels


    def predict_in_dataset(self, dataset, fold, stage, cb, data, limit=-1, batch_size=32, ttflips=False):
        with tqdm.tqdm(total=len(dataset), unit="files", desc="classification of images from " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                for i in range(len(b.data)):
                    id=b.data[i]
                    cb(id,b.results[i],data)
                pbar.update(batch_size)

    def predict_all_to_array(self, dataset, fold, stage, limit=-1, batch_size=32, ttflips=False):
        res=[]
        with tqdm.tqdm(total=len(dataset), unit="files", desc="classification of images from " + str(dataset)) as pbar:
            for v in self.predict_on_dataset(dataset, fold=fold, stage=stage, limit=limit, batch_size=batch_size, ttflips=ttflips):
                b=v
                for i in range(len(b.data)):
                    id=b.data[i]
                    res.append(b.results[i])
                pbar.update(batch_size)
        return np.array(res)

    def __inner_create(self, clazz, cleaned):
        mm=self.encoder_weights
        if mm is None:
            mm=self.weights
        if cleaned["input_shape"][2] != 3 and mm is not None and len(mm) > 0:
            if os.path.exists(self.path + ".mdl-nchannel"):
                cleaned["weights"] = None
                model = clazz(**cleaned)
                weightsPath = self.path + ".mdl-nchannel"
                model.load_weights(weightsPath)
                return model

            copy = cleaned.copy()
            copy["input_shape"] = (cleaned["input_shape"][0], cleaned["input_shape"][1], 3)
            model1 = clazz(**copy)
            cleaned["weights"] = None
            model = clazz(**cleaned)
            self.adaptNet(model, model1,self.copyWeights,cleaned["input_shape"][2])
            model.save_weights(self.path + ".mdl-nchannel")
            return model
        return clazz(**cleaned)

def parse(path) -> ClassificationPipeline:
    cfg = configloader.parse("classification", path)
    cfg.path = path
    return cfg

class ClassificationStage(generic.Stage):
    def unfreeze(self, model):
        for layer in model.layers:
            layer.trainable = True

        model.compile(model.optimizer, model.loss, model.metrics)

    def freeze(self, model):
        for layer in model.layers:
            if isinstance(layer, keras.layers.Dense):
                break

            layer.trainable = False

        model.compile(model.optimizer, model.loss, model.metrics)