

from dashboard.service.svn import svn_checkout

dest_path = svn_checkout(
    project="ENTTBL/MATERIAL/BE/trunk",
    username="ttornabene",
    password="TommasoLuglio2025."
)
print(f"Checked out to: {dest_path}")
