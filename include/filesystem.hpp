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
        bool found = false;
        findFile(*p, file, fullFilePath, found);
        return found;
    }

    void findFile(path filepath, const char *file, path *fullFilePath, bool &found){
        if(!found) {
            try {
                if (exists(filepath)) { // does filepath actually exist?
                    if (is_regular_file(filepath)) {
                        if (strcmp(filepath.filename().c_str(), file) == 0) {// is it the file we are looking for
                            *fullFilePath = filepath;
                            found = true;
                            return;
                        }
                    }else if (is_directory(filepath)){ // is p a directory?
                        for (directory_iterator p = directory_iterator(filepath); p != directory_iterator(); p++)
                            findFile(p->path(), file, fullFilePath, found);
                    }
                }
                else {
                    cout << "WARNING: " << filepath << " does not exist\n";
                    return;
                }
            }
            catch (const filesystem_error &ex) {
                cout << ex.what() << '\n';
            }
        }
    }

    void findDirectory(path filepath, const char *directory, path *fullDirectoryPath, bool &found){ // TODO: test this function
        if(!found) {
            try {
                if (exists(filepath)) { // does filepath actually exist?
                    if (is_directory(filepath)) {
                        for (directory_iterator p = directory_iterator(filepath); p != directory_iterator(); p++){
                            if (strcmp(p->path().filename().c_str(), directory) == 0) {// is it the directory we are looking for
                                *fullDirectoryPath = filepath;
                                found = true;
                                return;
                            }else{
                                findDirectory(p->path(), directory, fullDirectoryPath, found);
                            }
                        }
                    }else if (is_regular_file(filepath)){ // is p a file?
                        return;
                    }
                }
                else {
                    cout << "WARNING: " << filepath << " does not exist\n";
                    return;
                }
            }
            catch (const filesystem_error &ex) {
                cout << ex.what() << '\n';
            }
        }
    }

private:
    path *p;
};



