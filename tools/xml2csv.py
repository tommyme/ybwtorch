import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    count = 1
    for xml_file in glob.glob(path + '/*.xml'):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for member in root.findall('object'):
            try:
                if int(root.find('size')[0].text) == 0:
                  print('wrong size! no.',count,root.find('filename'))
                  continue
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                        )
            except TypeError:
                continue
                # print(count)
                # count+=1
                # value = (root.find('filename').text,
                #         #  int(root.find('size')[0].text),
                #         #  int(root.find('size')[1].text),
                #          member[0].text,
                #          int(member[2][0].text),
                #          int(member[2][1].text),
                #          int(member[2][2].text),
                #          int(member[2][3].text)
                #         )
            xml_list.append(value)
    column_name = ['filename','width','height','class_name','xmin','ymin','xmax','ymax']
    xml_df = pd.DataFrame(xml_list,columns = column_name)
    return xml_df
def main():
    image_path = os.path.join(os.getcwd(), 'Annotations')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('labels.csv',index=None)
    print('Successful!')

main()