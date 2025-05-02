inlets = 2

var div = 1.0;

function msg_float(f)
{
	if ((inlet==1))
		div = f
}

function list()
{
	var vals = arrayfromargs(arguments);
	
	for (i = 0; i < vals.length; i++)
		vals[i]=vals[i]/div;
		
	outlet(0,vals);

}