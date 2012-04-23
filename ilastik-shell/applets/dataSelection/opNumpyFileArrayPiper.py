from lazyflow.operators import OpArrayPiper

def createArrayPiperFromNpyFile(self, fileNames):
    """
    Open given .npy file(s) and produce an array piper operator with the data.
    """
    fileName = fileNames[0]
    if len(fileNames)>1:
        print "WARNING: only the first file will be read, multiple file prediction not supported yet"
    fName, fExt = os.path.splitext(str(fileName))
    raw = numpy.load(str(fileName))
    min, max = numpy.min(raw), numpy.max(raw)
    inputProvider = OpArrayPiper(self.graph)
    raw = raw.view(vigra.VigraArray)
    raw.axistags =  vigra.AxisTags(
        vigra.AxisInfo('t',vigra.AxisType.Time),
        vigra.AxisInfo('x',vigra.AxisType.Space),
        vigra.AxisInfo('y',vigra.AxisType.Space),
        vigra.AxisInfo('z',vigra.AxisType.Space),
        vigra.AxisInfo('c',vigra.AxisType.Channels))

    inputProvider.inputs["Input"].setValue(raw)
    return inputProvider

class OpNumpyFileArrayPiper(Operator):
    def __init__(self):
