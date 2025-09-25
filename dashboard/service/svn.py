import subprocess

def svn_checkout(project, username=None, password=None):
    repo_url = f"https://sqgate.lispa.local/repo/sw/{project}"
    dest_path = f"./volume_data/svn/{project}"
    
    cmd = ["svn", "checkout", repo_url, dest_path]
    print(f"Running command: {' '.join(cmd)}")
    
    if username and password:
        cmd.extend(["--username", username, "--password", password, "--non-interactive", "--trust-server-cert"])
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Checked out {repo_url} into {dest_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ SVN checkout failed: {e}")
    return dest_path;


# # from svn.remote import RemoteClient
# def svn_download(repo_url, dest_path):
#     r = RemoteClient(repo_url)
#     entries = r.list()  # list files in repo
#     print("Repo contents:", entries)

#     r.checkout(dest_path)
#     print(f"✅ Checked out {repo_url} into {dest_path}")

# # Example usage
# svn_download("https://svn.example.com/myproject/trunk", "./myproject")

# Example usage
# dest_path = svn_checkout(
#     project="ENTTBL/MATERIAL/BE/trunk",
#     username="ttornabene",
#     password="TommasoLuglio2025."
# )
# print(f"Checked out to: {dest_path}")