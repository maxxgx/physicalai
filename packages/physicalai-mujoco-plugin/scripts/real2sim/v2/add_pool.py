"""Insert the mixed-object pool (bucket_sort_objects.pool_xml) into a desk scene XML.
usage: add_pool.py IN_SCENE.xml OUT_SCENE.xml"""
import re, sys
from physicalai_mujoco_plugin.bucket_sort_objects import pool_xml

src, dst = sys.argv[1], sys.argv[2]
xml = open(src).read()
assets, bodies = pool_xml()
xml = re.sub(r"<!--BEGIN_POOL-->.*?<!--END_POOL-->", "", xml, flags=re.S)
xml = xml.replace("</asset>", "<!--BEGIN_POOL-->\n" + assets + "\n<!--END_POOL-->\n  </asset>", 1)
idx = xml.rindex("</worldbody>")
xml = xml[:idx] + "<!--BEGIN_POOL-->\n" + bodies + "\n<!--END_POOL-->\n  " + xml[idx:]
open(dst, "w").write(xml)
print("wrote", dst)
