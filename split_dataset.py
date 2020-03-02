#valset  = 0.2*trainset
#testset = 0.25*trainset
#so: train(80%) + val(20%) + test(20%) = fullset(100%)
#next_p = p/(1-p)

import os, random, shutil
import argparse


#command-line parameters
#-----------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--src_dir', type=str,required=True,
	help='Source directory'
)

parser.add_argument('--dest_dir', type=str,required=True,
	help='Destination directory'
)

parser.add_argument('--percentage', type=float,required=True,
	default=0.2,
	help='Destination directory'
)

FLAGS, unparsed = parser.parse_known_args()


SRC_DIR		=	FLAGS.src_dir
DEST_DIR	=	FLAGS.dest_dir

percentage_move = FLAGS.percentage

#-----------------------------------------------------------------------

if not os.path.exists( DEST_DIR ):
	os.mkdir( DEST_DIR )

for r, d, f in os.walk(SRC_DIR):
	directories = d
	break	#only first level

for r, directories, f in os.walk(SRC_DIR):

	for directory in directories:
		if not os.path.exists( os.path.join(DEST_DIR,directory) ):
			os.mkdir( os.path.join(DEST_DIR, directory) )
		
		print("Moving files from: "+directory)
		#walk files
		for s_r, s_d, files in os.walk( os.path.join(SRC_DIR, directory) ):
			files_num = len(files)
			for i in range( int(files_num*percentage_move) ):
				sel_file = random.choice(files)
				files.remove(sel_file)
				shutil.move( os.path.join(SRC_DIR, directory, sel_file),
							 os.path.join(DEST_DIR, directory, sel_file))
			
			print(str(int(files_num*percentage_move))+" files moved")
			break	#only first level
	
	break	#only first level

print("Files moved")
