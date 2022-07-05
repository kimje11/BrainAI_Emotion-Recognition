
import cv2
import numpy as np
from openvino.runtime import Core
import os

class OpvExec:
    __machine = []
    __machine_ids = []
    def __init__(self, machine_id=0):
        self._machine_id = machine_id
        if (not machine_id in self.__class__.__machine_ids):       #Create if it does not exist
            self.__class__.__machine_ids.append(machine_id)
            self.__class__.__machine.append(None)
    def _HasValidMachine(self):
        return (self._machine_id in self.__class__.__machine_ids) and self.__class__.__machine [self.__class__.__machine_ids.index(self._machine_id)] is not None
    def _SetMachine(self, machine):
        self.__class__.__machine[self.__class__.__machine_ids.index(self._machine_id)] = machine
    def _GetMachine(self):
        if (self._machine_id in self.__class__.__machine_ids):
            assert (self.__class__.__machine [self.__class__.__machine_ids.index(self._machine_id)] is not None), "Please check that a valid model has been loaded"
            return self.__class__.__machine [self.__class__.__machine_ids.index(self._machine_id)]

    #For releasing loaded graph on the device (e.g. NCS2)
    def ClearMachine(self):
        tmp = self.__class__.__machine[self.__class__.__machine_ids.index(self._machine_id)]
        self.__class__.__machine[self.__class__.__machine_ids.index(self._machine_id)] = None
        del tmp

class OpvModel(OpvExec):
    def __init__(self, model_name, device, fp, debug=False, ncs=1):
        OpvExec.__init__(self, ncs)
        assert(fp in ('FP16','FP32'))
        self.name = model_name
        self._debug = debug
        if (self._HasValidMachine()):
            self.ClearMachine()
            if (self._debug == True):
                print("Loaded Machine Released")

        #Load the .xml and .bin files for the model (and .labels if it exists)
        model_xml = "models/"+model_name+"/"+fp+"/"+model_name+".xml"
        model_bin = "models/"+model_name+"/"+fp+"/"+model_name+".bin"
        model_labels = os.path.splitext(model_xml)[0] + ".labels"
        if (os.path.exists(model_labels)):
            self.labels = np.loadtxt(model_labels,dtype="str",delimiter="\n")

        ie = Core()
        model = ie.read_model(model=model_xml)
        compiled_model = ie.compile_model(model=model, device_name="CPU")
        
        self.input_layer = compiled_model.input(0)
        self.input_shape = self.input_layer.shape
        self.output_layer =compiled_model.output(0)

        self._SetMachine(compiled_model)


    def Preprocess(self, image):                              # Preprocess the image
        original = image.copy()

        (n, c, h, w) = self.input_shape
        images = np.ndarray(shape=(n, c, h, w))
        if image.shape[:-1] != (h, w):
            if (self._debug):
                print("\t[INFO] Image resized from {} to {}".format(image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        images[0] = image.transpose((2, 0, 1))               # Change data layout from HWC to CHW
        return (original, images)

    def Predict(self, image, layer=None):
        if (layer==None):
            layer = self.output_layer
        (self.original,image) = self.Preprocess(image)

        self.lastresult = self._GetMachine().infer_new_request([image])
        output = self.lastresult[layer]
        return output                                        # Return the default output layer
