# -*- coding: utf-8 -*-

import numpy as np
from xml.dom.minidom import parse

def signalml_loader(file_name):
    '''
    It returns data and dictionary from SignalML files.
    
    Args:
      *file_name* : str
         must be the same for .xml and .raw files.
    Returns:
      *data*: np.array
         eeg data from raw file
      *xmlinfo* : dict
         dcitionary with keys: samplingFrequency, channelCount, firstSampleTimestamp,
         channelNames, calibrationCoef which means the same as in SML file
    '''
    raw_data = np.fromfile(file_name+'.raw','float32')
    xmlinfo=give_xml_info(file_name+'.xml')
    samp_cnt = xmlinfo['sampleCount']
    chan_cnt = xmlinfo['channelCount']
    data = np.zeros((chan_cnt,samp_cnt))
    for e,name in enumerate(xmlinfo['channelNames']):
        data[e] = raw_data[e::chan_cnt]
    return data, xmlinfo

def give_xml_info(path):
    '''
    It returns dictionary from SignalML file.
    
    Args:
      *path* : str
        SML file eg. 'test.xml'
    Returns:
      *xml_data* : dict
         dcitionary with keys: samplingFrequency, channelCount, firstSampleTimestamp,
         channelNames, calibrationCoef which means the same as in SML file
    '''
    
    try:
        doc=parse(path)
    except IOError or xml.parsers.expat.ExpatError:
        print 'Give a right path'
    info=['samplingFrequency','channelCount', 'firstSampleTimestamp','sampleCount']
    xml_data={}

    for nm in info:
        xml_data[nm] = float(doc.getElementsByTagName('rs:'+nm)[0].childNodes[0].data)

    chann_names,calibr_coef=[],[]
    chann_lab=doc.getElementsByTagName('rs:'+'channelLabels')[0]
    calibr_gain=doc.getElementsByTagName('rs:'+'calibrationGain')[0]
    for j in xrange(int(xml_data['channelCount'])):
        chann_names.append(chann_lab.getElementsByTagName('rs:'+'label')[j].childNodes[0].data)
        calibr_coef.append(float(calibr_gain.getElementsByTagName('rs:'+'calibrationParam')[j].childNodes[0].data))
    xml_data['channelNames'] = chann_names 
    xml_data['calibrationCoef'] = calibr_coef 

    return xml_data
