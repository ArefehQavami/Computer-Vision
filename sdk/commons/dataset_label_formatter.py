# Importing the required libraries
import xml.etree.ElementTree as Xet
import pandas as pd
import os


class DatasetFormatter:
    """
        A class to create labels in csv files.

                            Attributes:
                              None


                            Methods:
                              xml_to_csv_format(self, path: str)
                                A method to convert xml files to a unified csv file.

                            """
    def __init__(self):
        """
                Initializes class attributes.
                                    Args:
                                      None

                """
        pass

    def xml_to_csv_format(self, path: str):
        """This method convert all xml files to a csv file.

                            Args:
                              path
                                base path of xml files.
                            Returns:
                              None
                            Raises:
                              IOError: An error occurred reading xml files.
                            """
        cols = ["name", "xmin", "ymin", "xmax", "ymax", "label"]
        rows = []
        for filename in os.listdir(path):
            if not filename.endswith('.xml'): continue
            fullname = os.path.join(path, filename)
            xmlparse = Xet.parse(fullname)
            root = xmlparse.getroot()
            name = root.find("filename").text
            if root.find("object") is None:
                print(filename)
            xmin = root.find("object").find("bndbox").find("xmin").text
            ymin = root.find("object").find("bndbox").find("ymin").text
            xmax = root.find("object").find("bndbox").find("xmax").text
            ymax = root.find("object").find("bndbox").find("ymax").text
            label = root.find("object").find("name").text

            rows.append({"name": name,
                         "xmin": xmin,
                         "ymin": ymin,
                         "xmax": xmax,
                         "ymax": ymax,
                         "label": label
                         })

        df = pd.DataFrame(rows, columns=cols)
        labels = df.label.unique()
        extension = '.csv'
        for label in labels:
            df_with_specific_label = df.loc[df['label'] == label]
            df_with_specific_label.to_csv(f"dataset\\labels\\{label}{extension}", index=False, header=False)
