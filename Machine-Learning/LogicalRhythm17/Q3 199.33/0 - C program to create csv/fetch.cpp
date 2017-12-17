//Program to feach data from txt files

#include <bits/stdc++.h>
using namespace std;

//Structure for every training exapmle
struct data
{
	string elements[20];
}d;

//Function to print training exapmle
void print()
{
	int i;
	for(i=0;i<19;i++)
		cout<<d.elements[i]<<",";
	cout<<d.elements[i]<<endl;
}

//Function to reset Structure
void reset()
{
	for(int i=0;i<20;i++)
		d.elements[i] = "";
}

//Insert data to training exapmles at id location
void insert(string str, int id)
{
	d.elements[id] = str;
}

//Reach at first integer value in str from i location
int reach(string str, int i)
{
	while(!(str[i]>='0' && str[i]<='9'))
		i++;
	return i;
}


//Featch and return Id from str
string getId(string str)
{
	string s;
	int i;
	for(i = 0;str[i]!='6';i++);
	while(i<str.length())
	{
		s.push_back(str[i]);
		i++;
	}
	return s;
}


//Featch Date from str which is at location i
string getDate(string str, int i)
{
	string s;
	while(str[i-1] != 'M')
		{s.push_back(str[i]);i++;}
	return s;
}

//Featch float or int from str which is at location i
string fetch_float(string str,int i)
{
	string s;
	int c=0;
	while( ( (str[i]>='0' && str[i]<='9' ) || str[i] == '.' ) && !(str[i] == '.' && c == 1) )
	{
		if(str[i] == '.')
			c=1;
		s.push_back(str[i]);
		i++;
	}
	return s;
}

int main()
{
	//Add feature name in csv
	printf("House ID,Build Date,Priced Date,Garden Space,Dock,Capital,Royal,Guarding,River,Renovation,Dining Rooms,Bedromms,Bathrooms,Visit,Sorcerer,Blessings,In Front,Location,Holy Tree,Distance from Knight's house\n");

	//Open a txt file to featch data
	ifstream input("FileName.txt");

	string str;

	//Read data till we are not at EOF
	while(getline(input,str) && !input.eof())
	{
		//Vector of string to featch every feature details of training example
		vector<string> v;

		//Reset Structure
		reset();

		//Featch data of training example
		while(getline(input,str) && str.length() != 0)
		{
			v.push_back(str);
		}

		//Featch relevant information from text
		for (int i = 0; i < v.size(); ++i)
		{
			//If it is House ID then featch it
			if(v[i].find("House ID") != string::npos)
				insert(getId(v[i]),0);

			//If it is Built Date then featch it
			if(v[i].find("Built") != string::npos)
				insert(getDate(v[i],reach(v[i],0)),1);

			//If it is Priced Date then featch it
			if(v[i].find("Priced") != string::npos)
				insert(getDate(v[i],reach(v[i],v[i].length() - 21)),2);

			//If it is Garden detail then featch it
			if(v[i].find("garden") != string::npos)
				if(v[i].find("no") != string::npos)
				{
					string s = "2";
					insert(s,3);
				}
				else
				{
					string s = "1";
					insert(s,3);
				}

			//If it is Distance from Dock then featch it
			if(v[i].find("Dock") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),4);

			//If it is Distance from Capital then featch it
			if(v[i].find("Capital") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),5);

			//If it is Distance from Royal then featch it
			if(v[i].find("Royal") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),6);

			//If it is Distance from Guarding then featch it
			if(v[i].find("Guarding") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),7);

			//If it is Distance from River then featch it
			if(v[i].find("River") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),8);

			//If it is renovation detail then featch it
			if(v[i].find("renovation") != string::npos)
			{
				if(v[i].find("not") != string::npos)
				{
					string s = "2";
					insert(s,9);
				}
				else
				{
					string s = "1";
					insert(s,9);
				}
			}

			//If it is Dining rooms detail then featch it
			if(v[i].find("dining") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),10);

			//If it is Bedrooms detail then featch it
			if(v[i].find("bedrooms") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),11);

			//If it is Bathrooms detail then featch it
			if(v[i].find("bathrooms") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),12);

			//If it is kings visit detail then featch it
			if(v[i].find("visit") != string::npos || v[i].find("Visit") != string::npos)
				if(v[i].find("Visited") != string::npos)
				{
					string s = "2";
					insert(s,13);
				}
				else
				{
					string s = "1";
					insert(s,13);
				}

			//If it is Sorcerer detail then featch it
			if(v[i].find("orcerer") != string::npos)
			{
				if(v[i].find("Sorcerer") != string::npos)
				{
					string s = "2";
					insert(s,14);
				}
				else
				{
					string s = "1";
					insert(s,14);
				}
			}

			//If it is Blessing detail then featch it
			if(v[i].find("blessed") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),15);

			//If it is farm detail then featch it
			if(v[i].find("farm") != string::npos)
			{
				if(v[i].find("small land") != string::npos)
				{
					string s = "2";
					insert(s,16);
				}
				else if(v[i].find("huge land") != string::npos)
				{
					string s = "3";
					insert(s,16);
				}
				else
				{
					string s = "1";
					insert(s,16);
				}
			}

			//If it is Location of House then featch it
			if(v[i].find("Location") != string::npos)
			{
				if(v[i].find("The Mountains") != string::npos)
				{
					string s = "2";
					insert(s,17);
				}
				else if(v[i].find("Cursed Land") != string::npos)
				{
					string s = "3";
					insert(s,17);
				}
				else if(v[i].find("Servant's Premises") != string::npos)
				{
					string s = "4";
					insert(s,17);
				}
				else if(v[i].find("King's Landing") != string::npos)
				{
					string s = "5";
					insert(s,17);
				}
				else
				{
					string s = "1";
					insert(s,17);
				}
			}

			//If it is Holy Tree then featch it
			if(v[i].find("Holy tree") != string::npos)
				if(v[i].find("Ancient") != string::npos)
				{
					string s = "2";
					insert(s,18);
				}
				else
				{
					string s = "1";
					insert(s,18);
				}

			//If it is distance from Knight's house then featch it
			if(v[i].find("Knight's") != string::npos)
				insert(fetch_float(v[i],reach(v[i],0)),19);
		}
		print();
	}
}