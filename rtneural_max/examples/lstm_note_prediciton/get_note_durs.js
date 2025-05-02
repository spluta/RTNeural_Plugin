function list()
{
	var a = arrayfromargs(arguments);
	
	//comment this for loop out if you don't want the training to loop back to the beginning from the end of the list
	for (i = 0; i < a.length-1; i++) 
	{
		a[i]=a[i+1]-a[i];
	}
	a.pop();
	a.push(a[0]);
	
	min = 10000000;
	for (i = 0; i < a.length; i++) 
	{
		if(a[i]<min){min = a[i]};
	}
	
	for (i = 0; i < a.length; i++) 
	{
		a[i]=Math.round(a[i]/min);
		
	}
	outlet(0,a);
}