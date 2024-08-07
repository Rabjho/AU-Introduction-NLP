import pandas as pd
from pathlib import Path

def loadForumData(dir: str) -> pd.DataFrame:
    """
    Don't look inside, data is a mess -> code is too.
    Loads https://www.kaggle.com/datasets/ralphwiggins/hacker-forum-data/ data from directory (unzipped).
    """
    
    path = Path(dir)
    data = pd.DataFrame()
    for i, file in enumerate(path.glob('*.csv')):
        df = pd.read_csv(file)
        if (i == 0):
            # Clean up column names
            df.rename(columns={'PostNumber\n': 'PostNumber'}, inplace=True)
            
            # Fix date format
            df["PostDate"] = df["PostDate"].str.strip()
            # Remove ordinal suffixes
            df["PostDate"] = df["PostDate"].str.replace(r'(?<=\d)(st|nd|rd|th)', '', regex=True)
            df["PostDate"] = pd.to_datetime(df["PostDate"], format='%B %d, %Y, %I:%M %p')
            
        if (i == 1):
            df.drop(columns=['Unnamed: 9'], inplace=True)
            
            df["PostDate"] = df["PostDate"].str.strip()
            df["PostDate"] = pd.to_datetime(df["PostDate"], format='%d-%b-%y', errors='coerce')
            
        if (i == 2):
            # Remove artifact columns (all cols start with 'Unnamed')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Fix date format, first strip whitespace
            df["PostDate"] = df["PostDate"].str.strip()
            df["PostDate"] = pd.to_datetime(df["PostDate"], format='%d-%b-%y', errors='coerce')
            
        if (i == 3):
            df["PostDate"] = df["PostDate"].str.strip()
            df["PostDate"] = pd.to_datetime(df["PostDate"], format='%B %d, %Y, %I:%M%p')
        
        if (i in (0, 1, 2)):
            # Remove unnecessary columns
            df.drop(columns=["MemberType", "JoinDate", "NumberOfPosts"], inplace=True)
            
        if (i == 4):
            # Doesn't have year in date, so we ignore the file
            continue

        if (i == 5):
            # Remove unnecessary and artifacted columns
            df.drop(columns=['threadID', "URL", "postAuthorJoinDate", "contentWithHTMLTag"], inplace=True)
            
            # Clean up column names
            df.rename(columns={'postID': "ID", "authorName": "Username", "threadTitle": "ThreadTitle", "postDate": "PostDate", "flatContent": "PostContent", "postSequence": "PostNumber"}, inplace=True)
            
            df["PostDate"] = df["PostDate"].str.strip()
            df["PostDate"] = pd.to_datetime(df["PostDate"], format='%d.%m.%Y, %H:%M', errors='coerce')
            
        if (i in (4, 5)):
            # Reorder columns
            df = df[['ID', 'Username', 'PostDate', 'ThreadTitle', 'PostContent',
        'PostNumber']]
        
        # Set dtypes correctly
        df["ID"] = df["ID"].astype("string")
        df["Username"] = df["Username"].astype("string")
        df["ThreadTitle"] = df["ThreadTitle"].astype("string")
        df["PostContent"] = df["PostContent"].astype("string")
        df["PostNumber"] = pd.to_numeric(df["PostNumber"], errors='coerce')
        
        
        data = pd.concat([data, df], ignore_index=True)
        
    # Drop rows where PostContent or PostDate is missing or empty
    data.dropna(subset=['PostContent', 'PostDate'], inplace=True)
    data = data[data["PostContent"] != ""] 
    data = data[data["PostDate"].notnull()]
    data = data[~data["PostContent"].isnull()]
    
    print(data[data["PostContent"].isnull()]["PostContent"])
    
    # Drop duplicates
    data.drop_duplicates(subset=['PostContent'], inplace=True)
    
    return data