const fs = require('fs');
const express = require('express');
const port = 3000;
const app = express();

app.listen(port, (err) => {
    if(err){
        console.log(`Error Occured`);
    }else console.log(`Server is running on port ${port}`);
});