from segmentation_pipeline.impl import generic_config as generic
from segmentation_pipeline.impl import configloader
from segmentation_pipeline.impl import datasets
import keras.applications as apps
import os
import keras

custom_models={

}

class ClassificationPipeline(generic.GenericConfig):

    def  __init__(self,**atrs):
        super().__init__(**atrs)
        self.dataset_clazz =datasets.KFoldedDataSetImageClassification
        self.flipPred=False
        pass

    def createStage(self,x):
        return generic.Stage(x,self)

    def createNet(self):
        if self.architecture in custom_models:
            clazz=custom_models[self.architecture]
        else: clazz = getattr(apps, self.architecture)
        t: configloader.Type = configloader.loaded['classification'].catalog['ClassificationPipeline']
        r = t.custom()
        cleaned = {}
        for arg in self.all:
            pynama = t.alias(arg);
            if not arg in r:
                cleaned[pynama] = self.all[arg];
        if self.crops is not None:
            cleaned["input_shape"]=(cleaned["input_shape"][0]//self.crops,cleaned["input_shape"][1]//self.crops,cleaned["input_shape"][2])
        cleaned["include_top"]=False
        model1= self.innerCreate(clazz, cleaned)
        dl = keras.layers.Dense(self.all["classes"], activation=self.all["activation"])(model1.output);
        model = keras.Model(model1.input, dl);
        return model

    def innerCreate(self, clazz, cleaned):
        if cleaned["input_shape"][2] > 3 and self.encoder_weights != None and len(self.encoder_weights) > 0:
            if os.path.exists(self.path + ".mdl-nchannel"):
                cleaned["encoder_weights"] = None
                model = clazz(**cleaned)
                model.load_weights(self.path + ".mdl-nchannel")
                return model

            copy = cleaned.copy();
            copy["input_shape"] = (cleaned["input_shape"][0], cleaned["input_shape"][1], 3)
            model1 = clazz(**copy);
            cleaned["encoder_weights"] = None
            model = clazz(**cleaned)
            self.adaptNet(model, model1);
            model.save_weights(self.path + ".mdl-nchannel")
            return model
        return clazz(**cleaned)


def parse(path) -> ClassificationPipeline:
    cfg = configloader.parse("classification", path)
    cfg.path = path;
    return cfg