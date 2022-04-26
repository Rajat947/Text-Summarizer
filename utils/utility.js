const fs = require('fs');
module.exports.fileReadpromisify = function(url,path_to_project){
    // console.log(`${path_to_project}\\${url}`);
    return new Promise(function(resolve, reject){
        fs.readFile(url, (err) => {
            reject(err);
        }, (data) => {
            console.log(data);
            resolve(data);
        })
    });
}