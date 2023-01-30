import sklearn
from sklearn.metrics import cohen_kappa_score
import os

files = sorted(os.listdir("./kappa/"))
print(files)
print(len(files))

for f in files:
  if "_" in f:
    new_name = f.replace("_", "-")
    os.rename(os.path.join("./kappa/", f), os.path.join("./kappa/", new_name))
  if "–" in f:
    new_name = f.replace("–", "-")
    try:
      os.rename(os.path.join("./kappa/", f), os.path.join("./kappa/", new_name))
    except:
      pass

kappa = {}

for year in range(2009, 2022):
  print(year)
  kappa[year] = {}
  scores = []
  parsed_files = []
  for f in files:
    if f.startswith(str(year)):
      if f in parsed_files:
        continue
      if "ann1" in f:
        ann1 = f
        ann2 = f.split(".")[0] + ".ann2"
      else:
        ann2 = f
        ann1 = f.split(".")[0] + ".ann1"
      parsed_files.append(ann1)
      parsed_files.append(ann2)

      with open(os.path.join("./kappa/" + ann1), encoding="ISO-8859-1") as a1:
        try:
          annotations = [line.strip().split() for line in a1]
        except Exception as e:
          continue
        tags1 = []
        for ann in annotations:
          if len(ann) != 2:
            if ann != []:
              ann.append("O")
              #pass
            else:
              continue
          tags1.append(ann[1])

      with open(os.path.join("./kappa/" + ann2), encoding="ISO-8859-1") as a2:
        try:
          annotations = [line.strip().split() for line in a2]
        except Exception as e:
          #print(e, ann2)
          continue
        tags2 = []
        for ann in annotations:
          if len(ann) != 2:
            if ann != []:
              ann.append("O")
              #pass
            else:
              continue
          tags2.append(ann[1])

      while len(tags1) != len(tags2):
        if len(tags1) > len(tags2):
          if len(tags1) > len(tags2) + 2:
            print("the difference is too big", ann1, len(tags1), ann2, len(tags2))
            break
          tags2.append("O")
        else:
          if len(tags1) +2 < len(tags2):
            print("the difference is too big", ann1, len(tags1), ann2, len(tags2))
            break
          tags1.append("O")
      try:
        print("ok: ", ann1, ann2)
        scores.append(cohen_kappa_score(tags1, tags2))
      except Exception as e:
        print(e, ann1, ann2)
        print(tags1)
        print(tags2)
    #print(ann1, ann2)
  kappa[year] = scores

all = 0
for y in kappa.keys():
  print(y, len(kappa[y])," ", "\t", "{:.3f}".format(sum(kappa[y])/len(kappa[y])))
  all += sum(kappa[y])/len(kappa[y])
print(all/13)