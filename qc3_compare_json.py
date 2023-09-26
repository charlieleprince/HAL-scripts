#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:18:06 2023

@author: helium
"""


import tkinter as tk
from tkinter import filedialog
import json
import sys, os, datetime, random
import logging
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from HAL.gui.dataexplorer import getSelectionMetaDataFromCache
from pathlib import Path
import json
logger = logging.getLogger(__name__)

# /!\/!\/!\
# in order to be imported as a user script, two "global" variables
# have to be defined: NAME and CATEGORY
NAME = "Compare sequence parameters"  # display name, used in menubar and command palette
CATEGORY = "Qcontrol3"  # category (note that CATEGORY="" is a valid choice)



def find_diff_between_dicts(dict1, dict2):
    # Aplatisse les deux dictionnaires
    def flatten(d, parent_key='', sep='_'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    flat_dict1 = flatten(dict1)
    flat_dict2 = flatten(dict2)

    # Compare les dictionnaires aplatis
    diff_dict = {}
    for key in flat_dict1:
        if key in flat_dict2:
            if flat_dict1[key] != flat_dict2[key]:
                diff_dict[key] = {
                    'dict1': flat_dict1[key],
                    'dict2': flat_dict2[key]
                }
        else:
            diff_dict[key] = {
                'dict1': flat_dict1[key],
                'dict2': None
            }

    for key in flat_dict2:
        if key not in flat_dict1:
            diff_dict[key] = {
                'dict1': None,
                'dict2': flat_dict2[key]
            }

    return diff_dict




class JSONFileComparatorApp:
    def __init__(self, root,hal_main_window = None):
        self.root = root
        root.title("Comparaison de fichiers JSON")
        self._hal_main_window = hal_main_window
        
        self.default_folder = ""
        self.default_path = ""
        self.set_entry_file_initial_value()
        self.setup_ui()
        

    def setup_ui(self):
        # Créez des étiquettes et des boutons pour charger les fichiers
        self.label_file1 = tk.Label(self.root, text="Chemin du fichier JSON 1:")
        self.label_file1.pack()
        

        self.entry_file1 = tk.Entry(self.root, width=50)
        self.entry_file1.pack()

        self.button_load1 = tk.Button(self.root, text="Charger Fichier 1", command=self.load_file1)
        self.button_load1.pack()

        self.label_file2 = tk.Label(self.root, text="Chemin du fichier JSON 2:")
        self.label_file2.pack()

        self.entry_file2 = tk.Entry(self.root, width=50)
        self.entry_file2.pack()
        
        
        self.entry_file2.insert(0,self.default_path)
        self.entry_file1.insert(0,self.default_path)

        self.button_load2 = tk.Button(self.root, text="Charger Fichier 2", command=self.load_file2)
        self.button_load2.pack()

        # Bouton pour comparer les fichiers
        self.button_compare = tk.Button(self.root, text="Comparer les fichiers", command=self.compare_files)
        self.button_compare.pack()

        # Zone de texte pour afficher les résultats
        self.result_text = tk.Label(self.root, text="", wraplength=400)
        self.result_text.pack()

    def set_entry_file_initial_value(self):
        if self._hal_main_window is None:
                self.set_entry_file_default_value()
        else:
            # get metadata from current selection
            metadata = getSelectionMetaDataFromCache(self._hal_main_window,
                                                     update_cache=True)
            # -- get selected data
            selection = self._hal_main_window.runList.selectedItems()
            if not selection:
                self.set_entry_file_default_value()
                return
            else:
                data_class = self._hal_main_window.dataTypeComboBox.currentData()
                data = data_class()
                # get path
                item = selection[0]
                data.path = item.data(QtCore.Qt.UserRole)
                self.default_path = os.path.join(data.path.parent ,
                                                 "config",
                                                 "sequence_parameters.json"  ) 
                self.default_folder = os.path.join(data.path.parent , "config")
        self.check_initial_default_path()
        


    def check_initial_default_path(self):
        if (not os.path.exists(self.default_path)): # or os.path.exists(self.default_path)):
            print(f"Le chemin par défaut {self.default_path} n'esxiste pas.")
            self.default_path = ""
        

    def set_entry_file_default_value(self):
        now = datetime.datetime.now()
        if sys.platform.startswith('win'):
            self.default_path = os.path.join("E:", 
                                             now.strftime("%Y"), 
                                             now.strftime('%m'), 
                                             now.strftime('%d'))
            
        elif sys.platform.startswith('linux'):
            self.default_path = os.path.join("/mnt", 
                                             "manip_E", 
                                             now.strftime("%Y"), 
                                             now.strftime('%m'), 
                                             now.strftime('%d'))
        else:
            self.default_path  = ""
        self.default_folder = self.default_path


    def load_file1(self):
        file_path1 = filedialog.askopenfilename(filetypes=[("Fichiers JSON", "*.json"),
                                                           ("Fichiers JSON", "*.sequence_parameters")],
                                                initialdir = self.default_folder)
        self.entry_file1.delete(0, tk.END)
        self.entry_file1.insert(0, file_path1)

    def load_file2(self):
        file_path2 = filedialog.askopenfilename(filetypes=[("Fichiers JSON", "*.json")],
                                                initialdir = self.default_folder)
        self.entry_file2.delete(0, tk.END)
        self.entry_file2.insert(0, file_path2)

    def compare_files(self):
        file_path1 = self.entry_file1.get()
        file_path2 = self.entry_file2.get()

        try:
            with open(file_path1, 'r') as file1:
                data1 = json.load(file1)
            
            with open(file_path2, 'r') as file2:
                data2 = json.load(file2)
            diff_dict = find_diff_between_dicts(data1, data2)
            self.print_diff_dict(diff_dict)

        except FileNotFoundError:
            self.result_text.config(text="Erreur : Chemin de fichier non trouvé.")
        except json.JSONDecodeError as e:
            self.result_text.config(text=f"Erreur de décodage JSON : {e}")
        except Exception as e:
            self.result_text.config(text=f"Erreur : {e}")

    def print_diff_dict(self, diff_dict):
        Text =  "[Differences in parameter files]\n"
        if not diff_dict.items():
            print("Aucune diff.")
            try:
                Text = random.choice(CITATION_DIFFERENCE)
                print(Text)
                self._hal_main_window.metaDataText.setPlainText(Text)
            except:
                pass
        for key, value in diff_dict.items():
            Text += f"{key}\n"
            Text += f"File1: {value['dict1']}\n"
            Text += f"File2: {value['dict2']}\n"
            Text +=  "\n"
        if self._hal_main_window is not None:
            self._hal_main_window.metaDataText.setPlainText(Text)
        print(Text)
            
CITATION_DIFFERENCE = [
"""Être différent n'est ni une 
bonne ni une mauvaise chose. Cela 
signifie simplement que vous êtes 
suffisamment courageux pour être 
vous-même.

Albert Camus
""", 
"""
Tu es toi et je suis moi. Accepte-moi
tel que je suis. Ne cherche pas à 
dénaturer mon identité et ma 
civilisation.

Aimé Césaire
""",
"""
Il faut prendre conscience de l'apport 
d'autrui, d'autant plus riche que la 
différence avec soi-même est plus grande.

Albert Jacquard
 """,

"""
Accepter l'autre, c'est précisément l'accepter 
comme autre.

André Comte-Sponville

"""               ,
"""
Unifier, c'est nouer même les diversités 
particulières, non les effacer pour un 
ordre vain.    

Antoine de Saint-Exupéry

"""   
    ]



def main(hal_main_window):
    root = tk.Tk()
    app = JSONFileComparatorApp(root, hal_main_window = hal_main_window)
    root.mainloop()


