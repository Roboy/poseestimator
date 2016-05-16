#pragma once

#include <iostream>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
using namespace std;

class FileSystem{
public:
    FileSystem(const char* rootDirectory){
        p = new path(rootDirectory);
        try
        {
            if (exists(*p))    // does p actually exist?
            {
                if(!is_directory(*p))
                    cout << "WARNING: " << rootDirectory << " is not a directory" << endl;
            }
            else
                cout << p << " does not exist\n";
        }

        catch (const filesystem_error& ex)
        {
            cout << ex.what() << '\n';
        }
    }

    bool find(const char *file, path *fullFilePath){
        return find(*p, file, fullFilePath);
    }

    bool find(path filepath, const char *file, path *fullFilePath){
        try
        {
            if (exists(filepath)){    // does p actually exist?
                if (is_regular_file(filepath))        // is it the fiel we are looking for
                    if(strcmp(filepath.filename().c_str(),file)==0) {
                        *fullFilePath = filepath;
                        return true;
                    }else
                        return false;
                else if (is_directory(filepath))      // is p a directory?
                {
                    for(directory_iterator p=directory_iterator(filepath); p!=directory_iterator(); p++)
                        find(p->path(), file, fullFilePath);
                }
            }
            else
                cout << p << " does not exist\n";
        }
        catch (const filesystem_error& ex)
        {
            cout << ex.what() << '\n';
        }

    }

private:
    path *p;
};



