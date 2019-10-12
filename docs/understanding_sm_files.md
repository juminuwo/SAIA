Original source: https://github.com/stepmania/stepmania/wiki/sm

We will be using the songs matched with their sm files as training data.

Important fields:
(first versions are probably not going to worry about varying BPMS, or STOPS)

### \#OFFSET
Sets the offet between the beginning of the song and the start of the note data. 

### \#BPMS
Sets the BPMs for this song. BPMS are defined in the format `Beat=BPM`, with each value separated by a comma.

### \#STOPS
Sets the stops for this song. Stops are defined in the format `Beat=Seconds`, with each value separated by a comma.

###  \#NOTES
The Notes tag contains the following information:
 
* Chart type (e.g. `dance-single`)
* Description/author
* Difficulty (one of `Beginner`, `Easy`, `Medium`, `Hard`, `Challenge`, `Edit`)
* Numerical meter
* Groove radar values, generated by the program
* and finally, the note data itself.
 
The first five values are postfixed with a colon. Groove radar values are separated with commas.
 
Note data is defined in terms of "measures" where a measure is several lines of text, terminated by a comma. The final measure in a chart is terminated by a semicolon instead. Each line consists of a set of characters representing each playable column in the chart type.
 
Valid note types are 4th, 8th, 12th, 16th, 24th, 32nd, 48th, 64th, and 192nd. Each measure consists of a number of lines that corresponds to one of these numbers. The total number of beats covered by any given measure is 4, and each line represents a portion of that. If a measure has 64 lines, for example, each line represents 1/64th of those 4 beats, or 1/16th of a beat, with the first line representing beat 0 within the measure. The note type of a given line can be determined by taking said beat value, dividing by 4, and then simplifying the fraction as much as possible and looking at the denominator. If the denominator is 96, 192 is used as the note type instead.
 
### Note Values
These are the standard note values:
 
* `0` – No note
* `1` – Normal note
* `2` – Hold head
* `3` – Hold/Roll tail
* `4` – Roll head
* `M` – Mine