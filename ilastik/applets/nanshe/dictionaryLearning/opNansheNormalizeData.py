###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#		   http://ilastik.org/license.html
###############################################################################

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Oct 14, 2014 16:36:16 EDT$"



from lazyflow.graph import Operator, InputSlot, OutputSlot

import numpy

import nanshe
import nanshe.imp.segment
import nanshe.util.xnumpy


class OpNansheNormalizeData(Operator):
    """
    Given an input image and max/min bounds,
    masks out (i.e. sets to zero) all pixels that fall outside the bounds.
    """
    name = "OpNansheNormalizeData"
    category = "Pointwise"
    
    Input = InputSlot()

    Ord = InputSlot(value=2.0, stype="int")
    
    Output = OutputSlot()
    
    def __init__(self, *args, **kwargs):
        super( OpNansheNormalizeData, self ).__init__( *args, **kwargs )

    def setupOutputs(self):
        # Copy the input metadata to both outputs
        self.Output.meta.assignFrom( self.Input.meta )
    
    def execute(self, slot, subindex, roi, result):
        key = roi.toSlice()
        raw = self.Input[key].wait()
        raw = raw[..., 0]

        ord = self.Ord.value

        processed = raw.copy()
        processed.fill(0)

        processed[:, ~numpy.ma.getmaskarray(raw).max(axis=0)] = nanshe.imp.segment(
            nanshe.util.xnumpy.truncate_masked_frames(raw),
            **{"renormalized_images" : {
                "ord" : ord
            }
        }).reshape(len(raw), -1)

        processed = processed[..., None]
        
        if slot.name == 'Output':
            result[...] = processed

    def setInSlot(self, slot, subindex, roi, value):
        pass

    def propagateDirty(self, slot, subindex, roi):
        if slot.name == "Input":
            self.Output.setDirty(roi)
        elif slot.name == "Ord":
            self.Output.setDirty( slice(None) )
        else:
            assert False, "Unknown dirty input slot"
