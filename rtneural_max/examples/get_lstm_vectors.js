

var in_size = 5;
var out_size = 1;
var counter_start = 5;

outlets = 2;

function list()
{
	var a = arrayfromargs(arguments);
	
	//comment this for loop out if you don't want the training to loop back to the beginning from the end of the list
	for (i = 0; i < in_size; i++) 
	{
		a.push(a[i]);
	}
	var temp;
	//post("length " + a.length + "\n");
	for (i = 0; i < (a.length - in_size); i++) 
	{
		temp = a.slice(i, i+in_size);
		outlet(0,temp);
		outlet(1,a[i+in_size]);
	}
}

function anything()
{
	var a = arrayfromargs(messagename, arguments);
	if(a[0]=='set'){
		if(a[1]=='in_size'){
			in_size = a[2]
			}
		if(a[1]=='out_size'){
			out_size = a[2]
			}
		}
}